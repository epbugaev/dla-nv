import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.init_utils import get_padding


class MRFBlock(nn.Module): 
    """
    Multi-Receptive Field Fusion block. Includes multiple resnet blocks, each consisting of multiple dilated convolutions.
    """
    def __init__(self, channels, MRF_block_type): 
        super().__init__()

        if MRF_block_type == 1 or MRF_block_type == 2: 
            self.kernels = [3, 7, 11]
            self.dilations = [[[1, 1], [3, 1], [5, 1]]] * 3
            print('dilations:', self.dilations)
        else: 
            self.kernels = [3, 5, 7]
            self.dilations = [[[1], [2]], [[2], [6]], [[3], [12]]]

        self.convs = nn.ModuleList([])

        for kernel_id, kernel_size in enumerate(self.kernels): 
            current_dilations_blocks = self.dilations[kernel_id]
            for dilation_block in current_dilations_blocks:

                currrent_module_list = []
                for dilation in dilation_block:
                    currrent_module_list.append(nn.utils.weight_norm(
                        nn.Conv1d(channels, channels, kernel_size=kernel_size, dilation=dilation, padding=get_padding(kernel_size, dilation))))
                    
                self.convs.append(nn.Sequential(*currrent_module_list))

    def forward(self, x_real, **batch): 
        LeakyReLU = nn.LeakyReLU(0.1)

        for conv in self.convs: 
            x_real_new = LeakyReLU(x_real)
            x_real_new = conv(x_real_new)

            x_real = x_real + x_real_new

        return x_real


class GeneratorBlock(nn.Module): 
    def __init__(self, kernel_dim, channels_new, channels_old, stride, MRF_block_type): 
        super().__init__()
        
        self.conv_transpose = nn.utils.weight_norm(
            nn.ConvTranspose1d(channels_old, channels_new, kernel_size=kernel_dim, stride=stride, padding=(kernel_dim - stride) // 2))
        
        self.MRF_block = MRFBlock(channels_new, MRF_block_type=MRF_block_type)

    def forward(self, x_real, **batch): 
        LeakyReLU = nn.LeakyReLU(0.1)

        x_real = LeakyReLU(x_real)
        x_real = self.conv_transpose(x_real)
        x_real = self.MRF_block(x_real)
        return x_real


class Generator(nn.Module):
    def __init__(self, n_mels=80, kernels_combination=3, hidden_dim=256, MRF_block_type=3):
        super().__init__()
        self.kernels_combination = kernels_combination

        if self.kernels_combination == 1 or self.kernels_combination == 2: 
            self.conv_kernels = [16, 16, 4, 4]
        else: 
            self.conv_kernels = [16, 16, 8]

        self.convs = nn.ModuleList([])
        self.convs.append(nn.utils.weight_norm(nn.Conv1d(n_mels, hidden_dim, kernel_size=7, padding=get_padding(7, 1))))

        for id, kernel_dim in enumerate(self.conv_kernels): 
            self.convs.append(GeneratorBlock(kernel_dim, channels_new=hidden_dim // 2**(id+1), channels_old=hidden_dim // 2**id, stride=kernel_dim//2, MRF_block_type=MRF_block_type))

        LeakyReLU = nn.LeakyReLU(0.1)
        self.convs.append(LeakyReLU)
        self.convs.append(nn.utils.weight_norm(nn.Conv1d(hidden_dim // 2**(len(self.conv_kernels)), 1, kernel_size=7, padding=get_padding(7, 1))))

        self.convs.append(nn.Tanh())


    def forward(self, spectrogram, **batch):
        """
        Model forward method.

        Args:
            x_gen (Tensor): audio created by generator (batch_size, T).
            spectrogram (Tensor): real audio (batch_size, n_freq, T).
        Returns:
            score_gen (list): list of generated audio scores by each discriminator.
        """

        for conv in self.convs: 
            spectrogram = conv(spectrogram)

        return {'y_gen': spectrogram}
