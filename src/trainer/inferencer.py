import torch
import torchaudio
import pathlib
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.transforms.mel_spectrogram import MelSpectrogram
from speechbrain.pretrained import FastSpeech2

class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        generator,
        config,
        device,
        dataloaders,
        save_path,
        text_to_mel=False, 
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
        sample_rate=22050,
    ):
        """
        Initialize the Inferencer.

        Args:
            generator (nn.Module): PyTorch generator model for audio synthesis.
            config (DictConfig): Run config containing inferencer config.
            device (str): Device for tensors and model (e.g. 'cuda', 'cpu').
            dataloaders (dict[DataLoader]): Dataloaders for different sets of data.
            save_path (str): Path to save model predictions and other information.
            text_to_mel (bool): Whether to use text-to-mel generation with FastSpeech2.
            metrics (dict): Dict with metrics for inference (metrics[inference]). Each metric 
                is an instance of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): Transforms that should be applied on 
                the whole batch. Depend on the tensor name.
            skip_model_load (bool): If False, require the user to set pre-trained checkpoint
                path. Set this argument to True if the model desirable weights are defined
                outside of the Inferencer Class.
            sample_rate (int): Audio sample rate in Hz. Defaults to 22050.
        """
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device
        self.mel_spec = MelSpectrogram().to(device)
        self.sample_rate = sample_rate

        self.generator = generator
        self.discriminator_mpd = None
        self.discriminator_msd = None 

        self.text_to_mel = text_to_mel
        if self.text_to_mel:    
            self.text_to_mel_model = FastSpeech2.from_hparams(
                source="speechbrain/tts-fastspeech2-ljspeech",
            )
            self.text_to_mel_model.eval()

        self.batch_transforms = batch_transforms

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        # path definition

        print('new save_path:', save_path)
        self.save_path = save_path

        # define metrics
        self.metrics = metrics
        if self.metrics is not None and self.metrics["inference"] is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = None

        if not skip_model_load:
            # init model
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part):
        """
        Run batch through the model, compute metrics, and
        save predictions to disk.

        Save directory is defined by save_path in the inference
        config and current partition.

        Args:
            batch_idx (int): the index of the current batch.
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type
                of the partition (train or inference).
            part (str): name of the partition. Used to define proper saving
                directory.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform)
                and model outputs.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        if self.text_to_mel:
            print('batch[text]:', batch['text'])
            batch['spectrogram'] = self.text_to_mel_model(batch['text'][0])[0]
            print('new spectrogram:', batch['spectrogram'])
        else: # If it is filled with temporarty patch: calculate real mel-spec on GPU
            batch['spectrogram'] = self.mel_spec(batch['audio'])


        outputs = self.generator(**batch)
        batch.update(outputs)

        
        for gen_audio, audio_path in zip(
            batch['y_gen'], batch["audio_path"]
        ):
            if self.save_path is not None:
                # you can use safetensors or other lib here
                last_audio_part = audio_path.split("/")[-1]

                path = pathlib.Path(self.save_path) / (pathlib.Path(audio_path).stem + '.wav')
                print('saved:', path)
                torchaudio.save(path, gen_audio, self.sample_rate)

        return batch

    def _inference_part(self, part, dataloader):
        """
        Run inference on a given partition and save predictions

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """

        self.is_train = False
        self.generator.eval()

        if self.evaluation_metrics is not None:
            self.evaluation_metrics.reset() 

        # create Save dir
        if self.save_path is not None:
            pass
            # (self.save_path / part).mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                    metrics=self.evaluation_metrics,
                )

        if self.evaluation_metrics is None:
            return None
        return self.evaluation_metrics.result()
