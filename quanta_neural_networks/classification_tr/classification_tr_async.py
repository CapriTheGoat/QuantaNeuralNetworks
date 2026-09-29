import numpy as np
import torch
from einops import rearrange
from jaxtyping import Bool, Float
from torch import nn, Tensor
from torch.nn import functional as F
from loguru import logger
from quanta_neural_networks.integrator_batch_tr_async import PerPixelBayesian
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

    def forward(self, delta_cube, absolute_time_cube, photon_mask):
        # --- THE FIX: Sever recurrent memory between independent batches ---
        self.ssd.clear_hidden_state()
        
        if delta_cube.dim() == 4:
            b, h, w, t_raw = delta_cube.shape
        else:
            b = 1
            h, w, t_raw = delta_cube.shape
            delta_cube = delta_cube.unsqueeze(0)
            absolute_time_cube = absolute_time_cube.unsqueeze(0)
            photon_mask = photon_mask.unsqueeze(0)

        # 1. Process 4D surface
        x = self.integrator.process_delta_cube(
            delta_t_cube=delta_cube, 
            absolute_time_cube=absolute_time_cube, 
            photon_mask=photon_mask
        )
        
        t_index_ll = absolute_time_cube[0, 0, 0, :].cpu().numpy()
        subsampling = getattr(self.integrator, 'subsampling', 1)

        out = []
        for i in range(x.shape[-1]):
            x_step = x[..., i].unsqueeze(1)
            
            feat = F.relu(self.conv1(x_step))
            feat = F.relu(self.conv2(feat))

            flat_dim = feat.view(b, -1).shape[-1]
            pooled_dim = self.pool(feat).view(b, -1).shape[-1]
            
            if pooled_dim == getattr(self.ssd, 'in_dim', pooled_dim):
                feat = self.pool(feat).view(b, -1)
            elif flat_dim == getattr(self.ssd, 'in_dim', flat_dim):
                feat = feat.view(b, -1)
            else:
                feat = feat.view(b, -1) 

            time_idx = min((i + 1) * subsampling - 1, len(t_index_ll) - 1)
            time_instant = t_index_ll[time_idx]

            # SSD processes the spatial features
            feat = self.ssd.forward_online(feat, time_instant=time_instant)
            out.append(feat)

        # 2. Sequence aggregation
        stacked_out = torch.stack(out, dim=1)
        mean_feat = stacked_out.mean(dim=1) 
        
        return self.linear(mean_feat)
    
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

