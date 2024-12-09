import torch
import torch.nn as nn 
from src.transforms.mel_spectrogram import MelSpectrogram
import torch.nn.functional as F

class GeneratorLoss(nn.Module):
    """
    Adversarial loss function for Generator.
    """

    def __init__(self, features_loss_weight=2.0, mel_loss_weight=45.0):
        """
        Args:
            features_loss_weight (float): weight for the features loss.
            mel_loss_weight (float): weight for the mel-spectrogram loss.
        """
        super().__init__()
        self.features_loss_weight = features_loss_weight
        self.mel_loss_weight = mel_loss_weight

        self.mel_spec = MelSpectrogram()

    def discriminator_features_loss(self, features_gen, features_real):
        features_loss = 0
        for features_gen_block, features_real_block in zip(features_gen, features_real):
            for features_gen_tensor, features_real_tensor in zip(features_gen_block, features_real_block):
                features_loss += torch.mean(torch.abs(features_gen_tensor - features_real_tensor))

        return features_loss

    def generator_loss(self, discriminator_gen_outputs):
        loss = 0

        for ds_gen_tensor in discriminator_gen_outputs:
            loss += torch.mean(torch.square(1 - ds_gen_tensor))

        return loss

    def forward(self, msd_score_gen, mpd_score_gen, msd_features_gen, msd_features_real, mpd_features_gen, mpd_features_real, 
                y_gen, spectrogram, **batch):
        """
        Loss function calculation logic.

        Note that loss function must return dict. It must contain a value for
        the 'loss' key. If several losses are used, accumulate them into one 'loss'.
        Intermediate losses can be returned with other loss names.

        For example, if you have loss = a_loss + 2 * b_loss. You can return dict
        with 3 keys: 'loss', 'a_loss', 'b_loss'. You can log them individually inside
        the writer. See config.writer.loss_names.

        Args:
            msd_gen_outputs (Tensor): MSD discriminator outputs for generated samples.
            mpd_gen_output (Tensor): MPD discriminator outputs for generated samples.
            msd_features_gen (Tensor): MSD discriminator features for generated samples.
            msd_features_real (Tensor): MSD discriminator features for real samples.
            mpd_features_gen (Tensor): MPD discriminator features for generated samples.
            mpd_features_real (Tensor): MPD discriminator features for real samples.
            y_gen_mel (Tensor): mel-spectrogram for generated samples.
            y_real_mel (Tensor): mel-spectrogram for real samples.
        Returns:
            losses (dict): dict containing calculated loss functions.
        """

        discriminator_features_loss = (self.discriminator_features_loss(msd_features_gen, msd_features_real) + 
                                       self.discriminator_features_loss(mpd_features_gen, mpd_features_real)) * self.features_loss_weight

        generator_loss = self.generator_loss(msd_score_gen) + self.generator_loss(mpd_score_gen)

        #print('y_gen.shape:', y_gen.shape, 'spectrogram.shape:', spectrogram.shape, 'mel_spec(y_gen).shape:', self.mel_spec(y_gen).shape)
        mel_loss = F.l1_loss(self.mel_spec(y_gen)[:, :, :, :spectrogram.shape[-1]], spectrogram) * self.mel_loss_weight

        return {"full_generator_loss": discriminator_features_loss + generator_loss + mel_loss, 
                "discriminator_features_loss": discriminator_features_loss, 
                "generator_loss": generator_loss, 
                "mel_loss": mel_loss}

class DiscriminatorLoss(nn.Module):
    """
    Adversarial loss function for Discriminator.
    """

    def __init__(self):
        super().__init__()

    def discriminator_outputs_loss(self, discriminator_gen_outputs, discriminator_real_outputs, **batch):
        loss = 0

        for ds_gen_tensor, ds_real_tensor in zip(discriminator_gen_outputs, discriminator_real_outputs):
            loss += torch.mean(torch.square(1 - ds_real_tensor)) + torch.mean(torch.square(ds_gen_tensor))
            
        return loss

    def forward(self, discriminator_gen_outputs, discriminator_real_outputs, **batch):
        """
        Loss function calculation logic.

        Note that loss function must return dict. It must contain a value for
        the 'loss' key. If several losses are used, accumulate them into one 'loss'.
        Intermediate losses can be returned with other loss names.

        For example, if you have loss = a_loss + 2 * b_loss. You can return dict
        with 3 keys: 'loss', 'a_loss', 'b_loss'. You can log them individually inside
        the writer. See config.writer.loss_names.

        Args:
            discriminator_gen_outputs (Tensor): discriminator outputs for generated samples.
            discriminator_real_outputs (Tensor): discriminator outputs for real samples.
        Returns:
            losses (dict): dict containing calculated loss functions.
        """

        return {"discriminator_loss": self.discriminator_outputs_loss(discriminator_gen_outputs, discriminator_real_outputs)}
