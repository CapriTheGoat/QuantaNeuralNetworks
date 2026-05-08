import numpy as np
import torch
from einops import rearrange
from jaxtyping import Bool, Float
from torch import nn, Tensor
from torch.nn import functional as F
from loguru import logger
from quanta_neural_networks.integrator_batch import PerPixelBayesian
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
        if raw_photons.dim() == 4:
            b, height, width, t_raw = raw_photons.shape
        else:
            b = 1
            height, width, t_raw = raw_photons.shape
            raw_photons = raw_photons.unsqueeze(0)
        t_index_ll = np.arange(1, t_raw + 1)
        
        x = self.integrator.process_photon_cube(raw_photons, bocpd_gamma=bocpd_gamma, subsampling=self.subsampling)
        t_index_ll = t_index_ll[self.subsampling - 1 :: self.subsampling]

        b, h, w, t_sub = x.shape

        x = rearrange(x, 'b h w t -> (b t) 1 h w')
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        _, c, h_prime, w_prime = x.shape
        x = rearrange(x, '(b t) c h w -> t c (b h) w', b=b, t=t_sub)
        x, _ = self.ssd(x, t_index_ll)
        x = rearrange(x, 't c (b h) w -> t b c h w', b=b)
        x = rearrange(x, 't b c h w -> (t b) c h w')
        x = self.pool(x)
        x = rearrange(x, '(t b) c 1 1 -> t b c', b=b, t=t_sub)
        x = x.mean(dim=0)

        return self.linear(x)
    
    @torch.no_grad()
    def simulate_live_camera(self, raw_photons, bocpd_gamma: float = 1e-4):
        if raw_photons.dim() == 4:
            b, height, width, t_raw = raw_photons.shape
        else:
            b = 1
            height, width, t_raw = raw_photons.shape
            raw_photons = raw_photons.unsqueeze(0)

        t_index_ll = np.arange(1, t_raw + 1)
        
        x = self.integrator.process_photon_cube(raw_photons, bocpd_gamma=bocpd_gamma, subsampling=self.subsampling)
        t_index_ll = t_index_ll[self.subsampling - 1 :: self.subsampling]
        b, h, w, t_sub = x.shape

        x = rearrange(x, 'b h w t -> (b t) 1 h w')
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        _, c, h_prime, w_prime = x.shape
        
        x = rearrange(x, '(b t) c h w -> t c (b h) w', b=b, t=t_sub)
        
        self.ssd.clear_hidden_state()
        out_frames = []
        for e, t_index in enumerate(t_index_ll):
            frame = x[e] 
            frame_out = self.ssd.forward_online(frame, time_instant=t_index)
            out_frames.append(frame_out)
            
        x = torch.stack(out_frames, dim=0)
        
        x = rearrange(x, 't c (b h) w -> t b c h w', b=b)
        x = rearrange(x, 't b c h w -> (t b) c h w')
        x = self.pool(x)
        x = rearrange(x, '(t b) c 1 1 -> t b c', b=b, t=t_sub)
        x = x.mean(dim=0)
        
        return self.linear(x)

