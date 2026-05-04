import numpy as np
import torch
from einops import rearrange
from jaxtyping import Bool, Float
from torch import nn, Tensor
from torch.nn import functional as F
from loguru import logger
from quanta_neural_networks.integrator import PerPixelBayesian
from quanta_neural_networks.ssd import SSD



class BaselineClassifier(nn.Module):
    def __init__(self, subsampling=64, **kwargs):
        super().__init__()

        self.subsampling = subsampling

        # Integrator
        self.integrator = PerPixelBayesian(**kwargs)
        
        # 2D feature extractor
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        
        # Time Tracker
        self.ssd = SSD(in_dim=128, state_dim=12, head_dim=32)
        
        # Standard classifier head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(128, 10)
    def forward(self, raw_photons, bocpd_gamma: float = 1e-4):

        height, width, t = raw_photons.shape
        t_index_ll = np.arange(1, t + 1)
        x= self.integrator.process_photon_cube(raw_photons, bocpd_gamma=bocpd_gamma, subsampling=self.subsampling)
        t_index_ll = t_index_ll[self.subsampling - 1 :: self.subsampling]

        if x.dim() == 4:
            x = rearrange(x, 'b h w t -> (b t) 1 h w')
        elif x.dim() == 3:
            x = rearrange(x, 'h w t -> t 1 h w')
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x, _ = self.ssd(x, t_index_ll)
        x = self.pool(x).mean(dim=0)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.view(1, -1)
        return self.linear(x)

