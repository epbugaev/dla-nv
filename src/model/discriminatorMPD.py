import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.init_utils import get_padding


class DiscriminatorOnePeriod(nn.Module):
    """
    Discriminator for a single period K. Multiple of this module are stacked
    to form the Multi-Period Discriminator (DiscriminatorMPD).

    Notice that Conv2D here may look like Conv1D, but it is different according to
    Appendix C.1 of the paper in the way of sharing parameters.
    """

    def __init__(self, period: int):
        """
        Stride and kernel_size are set to parameters mentioned in the paper.

        Args:
            period (int): period K.
        """
        super().__init__()
        
        self.period = period

        self.conv_list = nn.ModuleList([])
        in_channels = 1
        for i in range(1, 5): 
            out_channels = 2**(5 + i)
            self.conv_list.append(nn.utils.weight_norm(nn.Conv2d(in_channels, out_channels, kernel_size=(5, 1), stride=(3, 1), padding=get_padding(5, 1))))
            in_channels = out_channels

        self.conv_list.append(nn.utils.weight_norm(nn.Conv2d(in_channels, in_channels, kernel_size=(5, 1), padding=get_padding(5, 1))))
        self.conv_list.append(nn.utils.weight_norm(nn.Conv2d(in_channels, 1, kernel_size=(3, 1), padding=get_padding(3, 1))))
        

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

        x = F.pad(x, (0, self.period - x.shape[-1] % self.period), "constant")
        x = x.view(x.size(0), x.size(1), -1, self.period) # Reshape to (batch_size, 1, T', K) 

        inter_features = []
        LeakyReLU = nn.LeakyReLU(0.1)
        for id, conv in enumerate(self.conv_list):
            x = conv(x)
            if id < len(self.conv_list) - 1:
                x = LeakyReLU(x)

            inter_features.append(x)

        return x.reshape(x.size(0), -1), inter_features


class DiscriminatorMPD(nn.Module):
    """
    Multi-Period Discriminator. Separates 1D audio into 2D matrix,
    where different rows correspond to different periods.
    """

    def __init__(self):
        """
        Args:
        """
        super().__init__()

        self.dms = nn.ModuleList([])

        self.periods = [2, 3, 5, 7, 11]
        for period in self.periods:
            self.dms.append(DiscriminatorOnePeriod(period))

    def forward(self, y_gen, y_real, **batch):
        """
        Model forward method.

        Args:
            x_gen (Tensor): audio created by generator (batch_size, T).
            x_real (Tensor): real audio (batch_size, 1, T).
        Returns:
            score_gen (list): list of generated audio scores by each discriminator.
            features_gen (list): list of generated audio features from each convolution layer by each discriminator.
            score_real (list): list of real audio scores by each discriminator.
            features_real (list): list of real audio features from each convolution layer by each discriminator.
        """

        score_gen, score_real = [], []
        features_gen, features_real = [], []
        
        for d in self.dms:
            score_real_i, features_real_i = d(y_real)
            score_gen_i, features_gen_i = d(y_gen)
            
            score_real.append(score_real_i)
            score_gen.append(score_gen_i)
            features_real.append(features_real_i)
            features_gen.append(features_gen_i)


        return {'mpd_score_gen': score_gen, 'mpd_features_gen': features_gen, 'mpd_score_real': score_real, 'mpd_features_real': features_real}
