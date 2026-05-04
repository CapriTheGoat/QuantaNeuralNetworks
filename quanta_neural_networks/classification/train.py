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
from quanta_neural_networks.classification.dataloader import IntensityCubeSimulatedNPY
from quanta_neural_networks.classification.classification import BaselineClassifier
from quanta_neural_networks.utils.hydra import print_and_save_cfg
from quanta_neural_networks.utils.train_utils import (
    resume_or_finetune,
    simulate_photon_cube,
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

    train_dataset = IntensityCubeSimulatedNPY(**cfg.data.train)
    val_dataset = IntensityCubeSimulatedNPY(**cfg.data.val)

    # Create dataloaders
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

    # Init model
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

    for epoch in range(epoch_start, cfg.num_epoch):
        logger.info(f"Train epoch {epoch + 1} | Global step {global_step}")
        with tqdm(total=len(train_dataset), dynamic_ncols=True) as pbar:
            model.train()
            for index, batch in enumerate(train_dataloader):
                target_label, photon_cube, intensity_ll = batch

                photon_cube = photon_cube.squeeze(0).to(device)
                target_label = target_label.to(device)

                logits = model.forward(photon_cube)

                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)
                
                loss = criterion(logits, target_label)

                accum_steps = cfg.model.get("gradient_accumulation_steps", 8)
                scaled_loss = loss / accum_steps
                scaled_loss.backward()

                if (index + 1) % accum_steps == 0:
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
                target_label, photon_cube, intensity_ll = batch

                photon_cube = photon_cube.squeeze(0).to(device)
                target_label = target_label.to(device)

                logits = model(photon_cube) 
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)

                loss = criterion(logits, target_label)
                total_val_loss += loss.item()

                predicted_class = torch.argmax(logits, dim=1)

                if predicted_class == target_label:
                    correct_predictions += 1
            
                total_samples += 1

                pbar.update(cfg.data.batch_size)
        
        avg_val_loss = total_val_loss / total_samples
        val_accuracy = (correct_predictions / total_samples) * 100

        print(f"Validation Loss: {avg_val_loss:.4f} | Validation Accuracy: {val_accuracy:.2f}%")

        writer.add_scalar("validation/loss", avg_val_loss, global_step=global_step)
        writer.add_scalar("validation/accuracy", val_accuracy, global_step=global_step)

        model.train()

if __name__ == "__main__":
    main()
