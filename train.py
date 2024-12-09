import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture, then print to console
    generator = instantiate(config.generator).to(device)
    discriminator_mpd = instantiate(config.discriminator_mpd).to(device)
    discriminator_msd = instantiate(config.discriminator_msd).to(device)

    logger.info(generator)
    logger.info(discriminator_mpd)
    logger.info(discriminator_msd)

    # get function handles of loss and metrics
    loss_generator = instantiate(config.loss_generator).to(device)
    loss_discriminator = instantiate(config.loss_discriminator).to(device)
    metrics = instantiate(config.metrics)

    # build optimizer, learning rate scheduler
    trainable_params_generator = filter(lambda p: p.requires_grad, generator.parameters())
    optimizer_generator = instantiate(config.optimizer_generator, params=trainable_params_generator)
    lr_scheduler_generator = instantiate(config.lr_scheduler_generator, optimizer=optimizer_generator)

    trainable_params_discriminator_mpd = filter(lambda p: p.requires_grad, discriminator_mpd.parameters())
    optimizer_discriminator_mpd = instantiate(config.optimizer_discriminator_mpd, params=trainable_params_discriminator_mpd)
    lr_scheduler_discriminator_mpd = instantiate(config.lr_scheduler_discriminator_mpd, optimizer=optimizer_discriminator_mpd)

    trainable_params_discriminator_msd = filter(lambda p: p.requires_grad, discriminator_msd.parameters())
    optimizer_discriminator_msd = instantiate(config.optimizer_discriminator_msd, params=trainable_params_discriminator_msd)
    lr_scheduler_discriminator_msd = instantiate(config.lr_scheduler_discriminator_msd, optimizer=optimizer_discriminator_msd)

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")

    trainer = Trainer(
        generator=generator,
        discriminator_mpd=discriminator_mpd,
        discriminator_msd=discriminator_msd,
        criterion_generator=loss_generator,
        criterion_discriminator=loss_discriminator,
        metrics=metrics,
        optimizer_generator=optimizer_generator,
        lr_scheduler_generator=lr_scheduler_generator,
        optimizer_discriminator_mpd=optimizer_discriminator_mpd,
        lr_scheduler_discriminator_mpd=lr_scheduler_discriminator_mpd,
        optimizer_discriminator_msd=optimizer_discriminator_msd,
        lr_scheduler_discriminator_msd=lr_scheduler_discriminator_msd,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()


if __name__ == "__main__":
    main()
