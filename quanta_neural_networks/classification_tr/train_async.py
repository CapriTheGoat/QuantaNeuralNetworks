"""
Training entrypoint for classification model.
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path

import hydra
import numpy as np
import torch
from einops import rearrange
from loguru import logger
from piq import ssim
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.optim as optim

from quanta_neural_networks.ssd import SSD
from quanta_neural_networks.ops.array_ops import loguniform
from quanta_neural_networks.ops.metrics import PSNR
from quanta_neural_networks.classification_tr.dataloader import TRMNISTAsynchronousDataset
from quanta_neural_networks.classification_tr.classification_tr_async import BaselineClassifier
from quanta_neural_networks.utils.hydra import print_and_save_cfg
from quanta_neural_networks.utils.train_utils import (
    resume_or_finetune,
    simulate_photon_cube,
)

def stream_to_delta_t_cube(event_stream: torch.Tensor, spatial_size=(28, 28), num_frames=100):
    # Ensure batched format [B, N, 3]
    if event_stream.dim() == 2:
        event_stream = event_stream.unsqueeze(0)
        
    B = event_stream.shape[0]
    dt_cubes, abs_time_cubes, photon_cubes = [], [], []
    
    for b in range(B):
        stream = event_stream[b]
        
        # Safely handle max time for batched/padded streams
        actual_max_time = stream[:, 2].max().item()
        if actual_max_time <= 0:
            actual_max_time = 1.0 
            
        frame_interval = actual_max_time / num_frames
        
        p_cube = torch.zeros((spatial_size[0], spatial_size[1], num_frames))
        d_cube = torch.zeros((spatial_size[0], spatial_size[1], num_frames))
        
        # Build the exact physical continuous time surface for this sample
        time_steps = torch.arange(1, num_frames + 1).float() * frame_interval
        a_cube = time_steps.view(1, 1, num_frames).expand(spatial_size[0], spatial_size[1], num_frames)
        
        last_spike_time = torch.zeros((spatial_size[0], spatial_size[1]))
        events = stream.cpu().numpy()
        
        for i in range(len(events)):
            y_raw, x_raw, t_raw = events[i]
            
            # THE FIX: Cast numpy.float32 to standard Python float
            t = float(t_raw)
            
            # Skip dataloader zero-padding artifacts
            if t == 0.0 and y_raw == 0.0 and x_raw == 0.0:
                continue
                
            y = min(max(int(y_raw) - 1, 0), spatial_size[0] - 1)
            x = min(max(int(x_raw) - 1, 0), spatial_size[1] - 1)
            
            frame_idx = min(int(t / frame_interval), num_frames - 1)
            
            # Use .item() to ensure clean float math
            dt = t - last_spike_time[y, x].item()
            
            p_cube[y, x, frame_idx] += 1.0
            d_cube[y, x, frame_idx] += dt
            last_spike_time[y, x] = t
            
        dt_cubes.append(d_cube)
        abs_time_cubes.append(a_cube)
        photon_cubes.append(p_cube)
        
    return torch.stack(dt_cubes), torch.stack(abs_time_cubes), torch.stack(photon_cubes)


if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

@hydra.main(
    config_path=f"../../conf",
    config_name=f"{Path(__file__).parent.name}_{Path(__file__).stem}",
    version_base="1.2",
)
def main(cfg):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device {device}")

    train_dataset = TRMNISTAsynchronousDataset(**cfg.data.train)
    val_dataset = TRMNISTAsynchronousDataset(**cfg.data.val)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        prefetch_factor=2,
    )
    val_dataloader = DataLoader(
        val_dataset, shuffle=True, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers
    )

    model = BaselineClassifier(**cfg.model.kwargs).to(device)

    for module in model.modules():
        if isinstance(module, SSD):
            module.parallel_mode = cfg.model.get("parallel_mode", False)
    
    ckpt_dir = Path(cfg.model.ckpt.folder)
    ckpt_dir.mkdir(exist_ok=True, parents=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), **cfg.optim)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.num_epoch * len(train_dataloader) // cfg.model.get("gradient_accumulation_steps", 8),
        **cfg.scheduler,
    )

    print_and_save_cfg(
        cfg,
        config_path_ll=[
            "config.yaml",
            Path(cfg.model.ckpt.folder) / "train_config.yaml",
        ],
    )

    logger.info(f"{len(train_dataset)} train samples, {len(val_dataset)} val samples.")
    Path(cfg.logging.tensorboard_dir).mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(str(cfg.logging.tensorboard_dir))

    epoch_start, global_step = resume_or_finetune(
        model, optimizer, cfg.model.ckpt, scheduler
    )
    epoch_start = global_step // len(train_dataset)

    # Train
    for epoch in range(epoch_start, cfg.num_epoch):
        logger.info(f"Train epoch {epoch + 1} | Global step {global_step}")
        with tqdm(total=len(train_dataset), dynamic_ncols=True) as pbar:
            model.train()
            for index, batch in enumerate(train_dataloader):
                target_label, event_stream = batch

                delta_cube, absolute_time_cube, photon_mask = stream_to_delta_t_cube(event_stream)

                delta_cube = delta_cube.to(device)
                absolute_time_cube = absolute_time_cube.to(device)
                photon_mask = photon_mask.to(device)
                target_label = target_label.to(device)

                logits = model.forward(delta_cube, absolute_time_cube, photon_mask)

                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)
                
                loss = criterion(logits, target_label)

                accum_steps = cfg.model.get("gradient_accumulation_steps", 8)
                scaled_loss = loss / accum_steps
                scaled_loss.backward()

                if (index + 1) % accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    
                if (index + 1) % 64 == 0:
                    scheduler.step()

                global_step += cfg.data.batch_size
                pbar.update(cfg.data.batch_size)

                if index % cfg.logging.scalar_interval == 0:
                    pbar.set_description(f"Train epoch {epoch + 1} | loss {loss.item():.3f}")
                    writer.add_scalar("training/loss", loss.item(), global_step=global_step)
                    
        if (epoch + 1) % cfg.model.ckpt.epoch_interval == 0:
            logger.info(f"Saving state to {ckpt_dir}")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "global_step": global_step,
                },
                ckpt_dir / f"checkpoint.pth",
            )

        # Validation
        model.eval() 
        total_val_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with tqdm(total=len(val_dataset), dynamic_ncols=True) as pbar, torch.no_grad():
            for index, batch in enumerate(val_dataloader):
                target_label, event_stream = batch

                delta_cube, absolute_time_cube, photon_mask = stream_to_delta_t_cube(event_stream)
                
                delta_cube = delta_cube.to(device)
                absolute_time_cube = absolute_time_cube.to(device)
                photon_mask = photon_mask.to(device)
                target_label = target_label.to(device)

                logits = model.forward(delta_cube, absolute_time_cube, photon_mask)
                
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)

                loss = criterion(logits, target_label)
                total_val_loss += loss.item() * target_label.size(0)

                predicted_class = torch.argmax(logits, dim=1)
                correct_predictions += (predicted_class == target_label).sum().item()
                total_samples += target_label.size(0)
                pbar.update(cfg.data.batch_size)
        
        avg_val_loss = total_val_loss / total_samples
        val_accuracy = (correct_predictions / total_samples) * 100

        print(f"Validation Loss: {avg_val_loss:.4f} | Validation Accuracy: {val_accuracy:.2f}%")

        writer.add_scalar("validation/loss", avg_val_loss, global_step=global_step)
        writer.add_scalar("validation/accuracy", val_accuracy, global_step=global_step)

        model.train()

if __name__ == "__main__":
    main()