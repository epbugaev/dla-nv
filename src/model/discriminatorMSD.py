import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.init_utils import get_padding


class DiscriminatorSingleScale(nn.Module):
    """
    Single version of MSD. Applies a single discriminator to an unscaled audio.
    """

    def __init__(self):
        super().__init__()
        self.conv_list = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(1, 16, kernel_size=15, stride=1, padding=get_padding(15, 1))),
            nn.utils.weight_norm(nn.Conv1d(16, 64, kernel_size=41, stride=4, padding=get_padding(41, 4))),
            nn.utils.weight_norm(nn.Conv1d(64, 256, kernel_size=41, stride=4, padding=get_padding(41, 4))),
            nn.utils.weight_norm(nn.Conv1d(256, 1024, kernel_size=41, stride=4, padding=get_padding(41, 4))),
            nn.utils.weight_norm(nn.Conv1d(1024, 1024, kernel_size=41, stride=4, padding=get_padding(41, 4))),
            nn.utils.weight_norm(nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=get_padding(5, 1))), 
            nn.utils.weight_norm(nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=get_padding(3, 1)))
        ])
        

    def forward(self, x, **batch):
        """
        Model forward method.

        Args:
            x (Tensor): input vector of shape (batch_size, 1, T).
        Returns:
            y (Tensor): discriminator logits.
            inter_features (Tensor): intermediate features from all conv layers.
        """

        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        inter_features = []
        LeakyReLU = nn.LeakyReLU(0.1) # Actually MelGAN uses IReLU, but for simplicity it is avoided
        for id, conv in enumerate(self.conv_list):
            x = conv(x)
            if id < len(self.conv_list) - 1:
                x = LeakyReLU(x)

            inter_features.append(x)

        return x.reshape(x.size(0), -1), inter_features



class DiscriminatorMSD(nn.Module):
    """
    Multi-Scale Discriminator. Applies three discriminators, each to more avg-pooled 
    versions of the input audio.
    """

    def __init__(self):
        """
        Args:
        """
        super().__init__()

        self.dms = nn.ModuleList([])

        for i in range(3):
            self.dms.append(DiscriminatorSingleScale())

    def forward(self, y_gen, y_real, **batch):
        """
        Model forward method.
        Before each next discriminator, audio is avg-pooled by 4.

        Args:
            x_gen (Tensor): generated audio (batch_size, 1, T).
            x_real (Tensor): real audio (batch_size, 1, T).
        Returns:
            score_gen (list): list of generated audio scores by each discriminator.
            features_gen (list): list of generated audio features from each convolution layer by each discriminator.
            score_real (list): list of real audio scores by each discriminator.
            features_real (list): list of real audio features from each convolution layer by each discriminator.
        """

        score_gen, score_real = [], []
        features_gen, features_real = [], []
        
        for id, d in enumerate(self.dms):
            if id > 0:
                y_gen = F.avg_pool1d(y_gen, kernel_size=4, stride=2)
                y_real = F.avg_pool1d(y_real, kernel_size=4, stride=2)

            score_real_i, features_real_i = d(y_real)
            score_gen_i, features_gen_i = d(y_gen)
            
            score_real.append(score_real_i)
            score_gen.append(score_gen_i)
            features_real.append(features_real_i)
            features_gen.append(features_gen_i)


        return {'msd_score_gen': score_gen, 'msd_features_gen': features_gen, 'msd_score_real': score_real, 'msd_features_real': features_real}
