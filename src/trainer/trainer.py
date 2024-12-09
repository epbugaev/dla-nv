from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.logger.utils import plot_spectrogram
import torch.nn.functional as F
import torch 

class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster
        batch['spectrogram'] = self.mel_spec(batch['audio'])

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer_discriminator_mpd.zero_grad()
            self.optimizer_discriminator_msd.zero_grad()

        y_gen = self.generator(**batch)
        y_gen_original = y_gen['y_gen']
        y_gen['y_gen'] = y_gen['y_gen'].detach() # Detach generator output while computing discriminator loss

        if y_gen['y_gen'].shape[-1] != batch['audio'].shape[-1]:
            batch['audio'] = F.pad(batch['audio'], (0, y_gen['y_gen'].shape[-1] - batch['audio'].shape[-1]), "constant")

        batch.update(y_gen)
        batch.update({'y_real': batch['audio']})

        msd_output = self.discriminator_msd(**batch)
        mpd_output = self.discriminator_mpd(**batch)

        batch.update(msd_output)
        batch.update(mpd_output)

        mpd_loss = self.criterion_discriminator(batch['mpd_score_gen'], batch['mpd_score_real'])
        msd_loss = self.criterion_discriminator(batch['msd_score_gen'], batch['msd_score_real'])
        total_discriminator_loss = mpd_loss['discriminator_loss'] + msd_loss['discriminator_loss']

        batch.update({'discriminator_loss': total_discriminator_loss})

        if self.is_train:
            batch['discriminator_loss'].backward()
            self._clip_grad_norm()
            self.optimizer_discriminator_mpd.step()
            self.optimizer_discriminator_msd.step()

            self.optimizer_generator.zero_grad()

            if self.lr_scheduler_discriminator_mpd is not None:
                self.lr_scheduler_discriminator_mpd.step()
            if self.lr_scheduler_discriminator_msd is not None:
                self.lr_scheduler_discriminator_msd.step()


        #y_gen = self.generator(**batch)
        y_gen['y_gen'] = y_gen_original
        print(y_gen['y_gen'].requires_grad)
        batch.update(y_gen)

        msd_output = self.discriminator_msd(**batch)
        mpd_output = self.discriminator_mpd(**batch)
        batch.update(msd_output)
        batch.update(mpd_output)
        
        generator_losses = self.criterion_generator(**batch)
        batch.update(generator_losses)

        if self.is_train:
            batch["full_generator_loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer_generator.step()
            
            if self.lr_scheduler_generator is not None:
                self.lr_scheduler_generator.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        if metric_funcs is not None:
            for met in metric_funcs:
                metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            # self.log_spectrogram(**batch)
            pass
        else:
            # Log Stuff
            self.log_audio(**batch)
            self.log_spectrogram(**batch)

    def log_audio(self, audio, y_gen, **batch):
        self.writer.add_audio("audio", audio, self.sample_rate)
        self.writer.add_audio("y_gen", y_gen.squeeze(1), self.sample_rate)

    def log_spectrogram(self, spectrogram, y_gen, **batch):
        gen_spectrogram = self.mel_spec(y_gen) + 1e-12
        original_spectrogram = spectrogram + 1e-12


        gen_spectrogram = gen_spectrogram.squeeze(1)[:, :, :original_spectrogram.shape[-1]]

        original_spectrogram_for_plot = original_spectrogram[0].detach().cpu()
        original_image = plot_spectrogram(original_spectrogram_for_plot)
        self.writer.add_image("original spectrogram", original_image)

        gen_spectrogram_for_plot = gen_spectrogram[0].detach().cpu()
        gen_image = plot_spectrogram(gen_spectrogram_for_plot)
        self.writer.add_image("generated spectrogram", gen_image)
