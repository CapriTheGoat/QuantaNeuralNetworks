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
    def __init__(self):
        super().__init__()
        # Integrator
        self.integrator = PerPixelBayesian(subsampling=64)
        
        # 2D feature extractor
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        
        # Time Tracker
        self.ssd = SSD(in_dim=128, state_dim=12, head_dim=32)
        
        # Standard classifier head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(128, 10)
    def forward(self, raw_photons):
        x = self.integrator.process_photon_cube(raw_photons)
        x = self.conv1(x)
        x = self.conv2(x)
        x, timestamps = self.ssd(x, timestamps)
        x = self.pool(x).mean(dim=0)
        return self.linear

