import numpy as np
import torch
from collections import deque
from einops import rearrange
from jaxtyping import Bool, Float
from torch import nn, Tensor
from torch.nn import functional as F
from loguru import logger
from quanta_neural_networks.integrator_batch_tr_test import PerPixelBayesian
from quanta_neural_networks.ssd_tr import SSD



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
        self.ssd = SSD(
            in_dim=128,
            state_dim=12,
            head_dim=32,
            subsampling=1,
        )
        
        # Standard classifier head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(128, 10)

    def forward(self, delta_cube, photon_mask, time_per_frame, bocpd_gamma: float = None):
        """
        TRAINING MODE
        """
        if delta_cube.ndim == 4:
            delta_cube = delta_cube.unsqueeze(0)
            photon_mask = photon_mask.unsqueeze(0)

        if delta_cube.ndim != 5 or photon_mask.shape != delta_cube.shape:
            raise ValueError(
                "Expected matching cubes with shape [B,H,W,T,K]."
            )

        b, height, width, t_raw, _ = delta_cube.shape
        
        t_scale = float(time_per_frame)

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
    def simulate_live_camera(self, event_stream):
        actual_max_time = event_stream[:, 2].max().item()
        
        # Match the 100-frame scale of training, and the 10-frame subsampling
        num_raw_frames = 100
        num_ssd_frames = num_raw_frames // self.subsampling
        readout_interval = actual_max_time / num_raw_frames
        
        self.integrator.update_hyperparams(bocpd_gamma=1e-4) 
        self.integrator.init_live_mode(h=28, w=28, device='cpu', time_per_frame=readout_interval)
        self.ssd.clear_hidden_state()
        device = next(self.parameters()).device
        
        events = event_stream.cpu().numpy()
        
        # We need exactly 10 snapshots (subsampling=10), spaced evenly in time
        next_snapshot_time = readout_interval * self.subsampling
        raw_snapshots = []
        ssd_times = []
        
        for i in range(len(events)):
            y, x, t = events[i]
            y_idx, x_idx = int(y) - 1, int(x) - 1
            
            # If physical time crosses the threshold, take a snapshot
            while t >= next_snapshot_time and len(raw_snapshots) < num_ssd_frames:
                snapshot = self.integrator.get_frame(current_t=next_snapshot_time)
                raw_snapshots.append(snapshot)
                ssd_times.append(next_snapshot_time)
                next_snapshot_time += (readout_interval * self.subsampling)
                
            self.integrator.process_single_event(y_idx, x_idx, float(t))
            
        # Catch remaining frames if the stream ends slightly early
        while len(raw_snapshots) < num_ssd_frames:
            snapshot = self.integrator.get_frame(current_t=next_snapshot_time)
            raw_snapshots.append(snapshot)
            ssd_times.append(next_snapshot_time)
            next_snapshot_time += (readout_interval * self.subsampling)

        # 1. Global Normalization (Fast, no CPU-GPU syncing mid-loop)
        raw_tensor = torch.stack(raw_snapshots, dim=0).to(device)
        global_max = raw_tensor.max().clamp(min=1e-6)
        normalized_tensor = (raw_tensor / global_max).clamp(0, 1)

        # 2. Synchronous Visual Cortex
        out_frames = []
        for i in range(len(normalized_tensor)):
            frame = normalized_tensor[i]
            feat = F.relu(self.conv1(frame))
            feat = F.relu(self.conv2(feat)).squeeze(0)
            
            # The SSD receives perfectly uniform physical time intervals
            frame_out = self.ssd.forward_online(feat, time_instant=ssd_times[i])
            out_frames.append(frame_out)
            
        stacked_frames = torch.stack(out_frames, dim=0) 
        pooled = self.pool(stacked_frames).squeeze(-1).squeeze(-1) 
        mean_feat = pooled.mean(dim=0).unsqueeze(0)

        return self.linear(mean_feat)

    
    def reset_camera(self, history_readouts):
        """Call once when starting a new camera stream."""
        if history_readouts < 1:
            raise ValueError("history_readouts must be positive.")

        self.ssd.clear_hidden_state()
        self._camera_first_packet = True
        self._camera_features = deque(maxlen=history_readouts)

    @torch.no_grad()
    def forward_camera(
        self,
        delta_cube,
        photon_mask,
        time_per_frame,
        current_time,
    ):
        """
        Process one NEW packet and emit one prediction.

        Each packet contains exactly self.subsampling raw time bins.
        """
        if (
            delta_cube.ndim != 5
            or delta_cube.shape[0] != 1
            or delta_cube.shape[-2] != self.subsampling
        ):
            raise ValueError(
                "Expected [1, H, W, self.subsampling, K]."
            )

        x = self.integrator.process_delta_cube(
            delta_cube,
            photon_mask,
            time_per_frame,
            subsampling=self.subsampling,
            normalize=True,
            clear_states=self._camera_first_packet,
        )
        self._camera_first_packet = False

        # One adaptive readout: [1, H, W, 1] -> [1, 1, H, W]
        x = x[..., 0].unsqueeze(1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # SSD retains its hidden state and previous timestamp.
        x = self.ssd.forward_online(
            x.squeeze(0),
            time_instant=float(current_time),
        )

        # [C, H', W'] -> [1, C]
        feature = self.pool(x.unsqueeze(0)).flatten(1)

        # Keep a bounded temporal window for a continuous camera.
        self._camera_features.append(feature)
        mean_feature = torch.stack(
            tuple(self._camera_features),
            dim=0,
        ).mean(dim=0)

        return self.linear(mean_feature)

