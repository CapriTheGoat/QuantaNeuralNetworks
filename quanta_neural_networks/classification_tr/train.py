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
import math

from quanta_neural_networks.ssd_tr import SSD
from quanta_neural_networks.ops.array_ops import loguniform
from quanta_neural_networks.ops.metrics import PSNR
from quanta_neural_networks.classification_tr.dataloader import TRMNISTAsynchronousDataset
from quanta_neural_networks.classification_tr.classification_tr import BaselineClassifier
from quanta_neural_networks.utils.hydra import print_and_save_cfg
from quanta_neural_networks.utils.train_utils import (
    resume_or_finetune,
    simulate_photon_cube,
)

def stream_to_delta_t_cube(
    event_stream: torch.Tensor,
    spatial_size=(28, 28),
    num_frames=100,
    max_time=1e-3,
    coordinate_base=1,
):
    """Return interarrival times and binary masks: [1,H,W,T,K]."""
    if event_stream.dim() == 3:
        if event_stream.shape[0] != 1:
            raise ValueError("Set data.batch_size=1 for this converter.")
        event_stream = event_stream[0]

    if event_stream.ndim != 2 or event_stream.shape[1] != 3:
        raise ValueError("Expected events with shape [N,3]: (y,x,t).")
    if coordinate_base not in (0, 1):
        raise ValueError("coordinate_base must be 0 or 1.")
    if num_frames < 1 or max_time <= 0:
        raise ValueError("num_frames and max_time must be positive.")

    H, W = spatial_size
    device = event_stream.device
    events = event_stream.to(torch.float64)

    if not torch.isfinite(events).all():
        raise ValueError("Events must contain finite coordinates and times.")

    events = events[torch.argsort(events[:, 2], stable=True)]

    xy = events[:, :2] - coordinate_base
    if (xy != xy.round()).any():
        raise ValueError("Pixel coordinates must be integers.")

    y, x = xy[:, 0].long(), xy[:, 1].long()
    t = events[:, 2]

    if ((y < 0) | (y >= H) | (x < 0) | (x >= W)).any():
        raise ValueError("Check coordinate_base and sensor dimensions.")
    boundary_tolerance = 1e-9 

    too_early = t < -boundary_tolerance
    too_late = t > max_time + boundary_tolerance

    if (too_early | too_late).any():
        raise ValueError(
            "Timestamp range exceeds boundary tolerance: "
            f"t_min={t.min().item():.17g}, "
            f"t_max={t.max().item():.17g}, "
            f"max_time={max_time:.17g}, "
            f"too_early={too_early.sum().item()}, "
            f"too_late={too_late.sum().item()}"
        )

    t = t.clamp(min=0.0, max=max_time)

    time_per_frame = max_time / num_frames
    frame_indices = (t / time_per_frame).floor().long()
    frame_indices = frame_indices.clamp(max=num_frames - 1)

    # Counts are used ONLY to allocate enough event slots.
    cell = (y * W + x) * num_frames + frame_indices
    occupancy = torch.bincount(cell, minlength=H * W * num_frames)
    K = max(1, int(occupancy.max().item()))

    delta_t_cube = torch.zeros(
        (H, W, num_frames, K), device=device, dtype=torch.float64
    )
    photon_mask = torch.zeros_like(
        delta_t_cube, dtype=torch.float32
    )
    order = torch.argsort(cell, stable=True)

    ordered_cells = cell[order]
    ordered_t = t[order]
    ordered_pixels = ordered_cells // num_frames

    # Previous arrival at the same pixel, including across bin boundaries.
    previous_t = torch.zeros_like(ordered_t)
    previous_t[1:] = torch.where(
        ordered_pixels[1:] == ordered_pixels[:-1],
        ordered_t[:-1],
        0.0,
    )

    interarrival_times = ordered_t - previous_t

    # Position of each photon within its pixel/bin group.
    starts = occupancy.cumsum(dim=0) - occupancy
    slots = (
        torch.arange(len(events), device=device)
        - starts[ordered_cells]
    )

    delta_t_cube.view(-1, K)[ordered_cells, slots] = interarrival_times
    photon_mask.view(-1, K)[ordered_cells, slots] = 1.0

    return (
        delta_t_cube.unsqueeze(0),
        photon_mask.unsqueeze(0),
        time_per_frame,
    )

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

@hydra.main(
    config_path=f"../../conf",
    config_name=f"{Path(__file__).parent.name}_{Path(__file__).stem}",
    version_base="1.2",
)

def main (cfg):
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
    accum_steps = cfg.model.get("gradient_accumulation_steps", 8)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.num_epoch * math.ceil(
            len(train_dataloader) / accum_steps
        ),
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
            print(
    "Integrator settings:",
    f"normalize={model.integrator.normalize},",
    f"quantile={model.integrator.quantile},",
    f"subsampling={model.subsampling}",
)
            optimizer.zero_grad()
            for index, batch in enumerate(train_dataloader):
                target_label, event_stream = batch

                delta_cube, photon_mask, time_per_frame= stream_to_delta_t_cube(event_stream)

                delta_cube = delta_cube.to(device)
                photon_mask = photon_mask.to(device)
                target_label = target_label.to(device)

                logits = model.forward(delta_cube, photon_mask, time_per_frame)

                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)
                
                loss = criterion(logits, target_label)

                group_start = (index // accum_steps) * accum_steps
                group_size = min(
                    accum_steps,
                    len(train_dataloader) - group_start,
                )

                (loss / group_size).backward()

                if (
                    (index + 1) % accum_steps == 0
                    or index + 1 == len(train_dataloader)
                ):
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()


                global_step += cfg.data.batch_size
                pbar.update(cfg.data.batch_size)

                if index % cfg.logging.scalar_interval == 0:
                    pbar.set_description(
                        f"Train epoch {epoch + 1} | loss {loss.item():.3f}"
                    )

                    writer.add_scalar(
                        "training/loss", loss.item(), global_step=global_step
                    )
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

        with tqdm(
            total=len(val_dataset), dynamic_ncols=True
        ) as pbar, torch.no_grad():
            for index, batch in enumerate(val_dataloader):
                target_label, event_stream = batch

                delta_cube, photon_mask, time_per_frame= stream_to_delta_t_cube(event_stream)
                
                delta_cube = delta_cube.to(device)
                photon_mask = photon_mask.to(device)
                target_label = target_label.to(device)

                logits = model.forward(delta_cube, photon_mask, time_per_frame)
                
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)

                loss = criterion(logits, target_label)
                total_val_loss += loss.item() * target_label.size(0)

                predicted_class = torch.argmax(logits, dim=1)

                correct_predictions += (predicted_class == target_label).sum().item()
            
                total_samples += target_label.size(0)

                predicted_class = torch.argmax(logits, dim=1)

                pbar.update(cfg.data.batch_size)
        
        avg_val_loss = total_val_loss / total_samples
        val_accuracy = (correct_predictions / total_samples) * 100

        print(f"Validation Loss: {avg_val_loss:.4f} | Validation Accuracy: {val_accuracy:.2f}%")

        writer.add_scalar("validation/loss", avg_val_loss, global_step=global_step)
        writer.add_scalar("validation/accuracy", val_accuracy, global_step=global_step)

        model.train()

if __name__ == "__main__":
    main()
