import numpy as np
import torch
from einops import rearrange
from jaxtyping import Bool, Float
from torch import nn, Tensor
from torch.nn import functional as F
from loguru import logger
from quanta_neural_networks.integrator_batch_tr_test import PerPixelBayesian
from quanta_neural_networks.ssd import SSD



class BaselineClassifier(nn.Module):
    def __init__(self, subsampling=10, **kwargs):
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

    def forward(self, delta_cube, photon_mask, time_per_frame, bocpd_gamma: float = 1e-4):
        """
        TRAINING MODE
        """
        if delta_cube.dim() == 4:
            b, height, width, t_raw = delta_cube.shape
        else:
            b = 1
            height, width, t_raw = delta_cube.shape
            delta_cube = delta_cube.unsqueeze(0)
            photon_mask = photon_mask.unsqueeze(0)
            
        if delta_cube.dim() == 4:
            b, height, width, t_raw = delta_cube.shape
        else:
            b = 1
            height, width, t_raw = delta_cube.shape
            delta_cube = delta_cube.unsqueeze(0)
            photon_mask = photon_mask.unsqueeze(0)
            
        t_scale = time_per_frame[0].item() if isinstance(time_per_frame, torch.Tensor) else float(time_per_frame)
        
        t_index_ll = (np.arange(1, t_raw + 1) * t_scale).astype(np.float32)
        
        # 1. Continuous-time state update
        x = self.integrator.process_delta_cube(
            delta_cube, 
            photon_mask,
            time_per_frame,
            bocpd_gamma=bocpd_gamma, 
            subsampling=self.subsampling,
            normalize=True
        )

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
        x = rearrange(x, 't c (b h) w -> (t b) c h w', b=b)
        x = self.pool(x)
        
        x = rearrange(x, '(t b) c 1 1 -> t b c', b=b, t=t_sub)
        
        x = x.mean(dim=0)

        return self.linear(x)
    
    @torch.no_grad()
    def simulate_live_camera(self, event_stream, max_time: float, num_training_frames: int = 64):
        self.integrator.init_live_mode(h=28, w=28, device='cpu')
        self.ssd.clear_hidden_state()
        device = next(self.parameters()).device
        
        events = event_stream.cpu().numpy()
        
        num_raw_frames = 100 
        readout_interval = max_time / num_raw_frames
        next_readout_time = readout_interval
        
        raw_frames = []
        frame_counter = 1
        
        for i in range(len(events)):
            y, x, t = events[i]
            y_idx, x_idx = int(y) - 1, int(x) - 1
            
            while t >= next_readout_time:
                # ONLY save the frame if it matches the training subsampling (e.g., every 10th frame)
                if frame_counter % self.subsampling == 0:
                    snapshot = self.integrator.get_frame(current_t=next_readout_time)
                    raw_frames.append(snapshot)
                
                frame_counter += 1
                next_readout_time += readout_interval
                
            self.integrator.process_single_event(y_idx, x_idx, float(t))
            
        # Catch up any remaining frames if the stream ends slightly before max_time
        while frame_counter <= num_raw_frames:
            if frame_counter % self.subsampling == 0:
                snapshot = self.integrator.get_frame(current_t=next_readout_time)
                raw_frames.append(snapshot)
            frame_counter += 1
            next_readout_time += readout_interval
            
        # 4. Global Normalization (Matches `clamp_recons`)
        raw_tensor = torch.stack(raw_frames, dim=0).to(device) # Shape: [10, 1, 1, 28, 28]
        global_max = raw_tensor.max().clamp(min=1e-6)
        normalized_tensor = (raw_tensor / global_max).clamp(0, 1)
        
        # 5. Synchronous Visual Cortex: Process the 10 normalized frames
        out_frames = []
        time_index = self.subsampling
        
        for i in range(len(normalized_tensor)):
            frame = normalized_tensor[i]
            
            feat = F.relu(self.conv1(frame))
            feat = F.relu(self.conv2(feat)).squeeze(0)
            
            frame_out = self.ssd.forward_online(feat, time_instant=float(time_index))
            out_frames.append(frame_out)
            
            time_index += self.subsampling
            
        # 6. Classifier Head
        stacked_frames = torch.stack(out_frames, dim=0) 
        pooled = self.pool(stacked_frames).squeeze(-1).squeeze(-1) 
        mean_feat = pooled.mean(dim=0).unsqueeze(0)

        return self.linear(mean_feat)

