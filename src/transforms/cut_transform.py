import torch
from torch import nn
import random

class CutTransform(nn.Module):
    """
    Batch-version of Normalize for 1D Input.
    Used as an example of a batch transform.
    """

    def __init__(self, use_random=False, cut_length=2, sample_rate=22050):
        """
        Args:
            mean (float): mean used in the normalization.
            std (float): std used in the normalization.
        """
        super().__init__()

        self.use_random = use_random
        self.cut_length = cut_length
        self.sample_rate = sample_rate  
        self.random_generator = random.Random(42)

    def forward(self, x):
        """
        Args:
            x (Tensor): input tensor.
        Returns:
            x (Tensor): normalized tensor.
        """
        target_len = self.cut_length * self.sample_rate
        if x.shape[-1] < target_len:
            return x

        if self.use_random:
            start = self.random_generator.randint(0, x.shape[-1] - target_len - 1)
            x = x[:, start:start+target_len]
        else:
            x = x[:, :target_len]


        return x

