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
from quanta_neural_networks.classification.dataloader import IntensityCubeSimulatedNPYMorphed
from quanta_neural_networks.classification.dataloader import stochastic_spad_morph
from quanta_neural_networks.classification.classification_morphed_priority import BaselineClassifier
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

    train_dataset = IntensityCubeSimulatedNPYMorphed(**cfg.data.train)
    val_dataset = IntensityCubeSimulatedNPYMorphed(**cfg.data.val)

    
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        #prefetch_factor=2,
        drop_last=True
    )
    val_dataloader = DataLoader(
        val_dataset, shuffle=True, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers, drop_last=True
    )

    import torch._inductor.config as inductor_config
    inductor_config.layout_optimization = False
    
    model = BaselineClassifier(**cfg.model.kwargs).to(device)

    model = torch.compile(model)

    for module in model.modules():
        if isinstance(module, SSD):
            module.parallel_mode = cfg.model.get("parallel_mode", False)
    
    ckpt_dir = Path(cfg.model.ckpt.folder)
    ckpt_dir.mkdir(exist_ok=True, parents=True)

    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
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
                break
                label_A, label_B, cube_A, cube_B = batch

                cube_A = cube_A.to(device)
                cube_B = cube_B.to(device)
                label_A = label_A.to(device)
                label_B = label_B.to(device)

                morphed_cube = stochastic_spad_morph(cube_A, cube_B)
                morphed_cube = morphed_cube.contiguous()

                T = morphed_cube.shape[3]
                target_label = torch.full((cube_A.shape[0], T), -100, dtype=torch.long, device=device)
                
                mid = T // 2
                half_wipe = 100 // 2
                
                target_label[:, :mid - half_wipe] = label_A.unsqueeze(1)
                target_label[:, mid + half_wipe:] = label_B.unsqueeze(1)

                # Subsample to match the dataloader/model setting
                subsampling = 20
                target_label = target_label[:, ::subsampling]

                logits = model.forward(morphed_cube)

                logits = logits.permute(1, 2, 0).contiguous()

                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)
                
                loss = criterion(logits, target_label)

                accum_steps = cfg.model.get("gradient_accumulation_steps", 8)
                scaled_loss = loss / accum_steps
                scaled_loss.backward()

                if (index + 1) % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                if (index + 1) % 64 == 0:
                    scheduler.step()


                global_step += cfg.data.batch_size
                pbar.update(cfg.data.batch_size)

                if index % cfg.logging.scalar_interval == 0:
                    pbar.set_description(
                        f"Train epoch {epoch + 1} | loss {loss.item():.3f}"
                    )

                    writer.add_scalar(
                        "training/loss", loss.item(), global_step=global_step
                    )
                    writer.add_scalar("training/learning_rate", optimizer.param_groups[0]['lr'], global_step=global_step)
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
        
        total_start_correct = 0
        total_start_samples = 0
        total_end_correct = 0
        total_end_samples = 0
        total_final_frame_correct = 0

        frame_by_frame_correct = None
        frame_by_frame_valid = None

        with tqdm(
            total=len(val_dataset), dynamic_ncols=True
        ) as pbar, torch.no_grad():
            for index, batch in enumerate(val_dataloader):
                label_A, label_B, cube_A, cube_B = batch

                cube_A = cube_A.to(device)
                cube_B = cube_B.to(device)
                label_A = label_A.to(device)
                label_B = label_B.to(device)

                morphed_cube = stochastic_spad_morph(cube_A, cube_B)
                morphed_cube = morphed_cube.contiguous()

                T = morphed_cube.shape[3]
                target_label = torch.full((cube_A.shape[0], T), -100, dtype=torch.long, device=device)
                
                mid = T // 2
                
                target_label[:, :mid] = label_A.unsqueeze(1)
                target_label[:, mid:] = label_B.unsqueeze(1)

                # Subsample to match the dataloader/model setting
                subsampling = 20
                target_label = target_label[:, ::subsampling]

                logits = model(morphed_cube) 
                logits = logits.permute(1, 2, 0).contiguous()
                
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)

                loss = criterion(logits, target_label)
                total_val_loss += loss.item()

                predicted_class = torch.argmax(logits, dim=1) # Shape: [Batch, Time]


                T = target_label.shape[1]
            
                valid_mask = target_label != -100
                correct_predictions += ((predicted_class == target_label) & valid_mask).sum().item()
                total_samples += valid_mask.sum().item()

                if frame_by_frame_correct is None:
                    # Initialize tensors of shape [Time] on the first batch
                    frame_by_frame_correct = torch.zeros(T, device=device)
                    frame_by_frame_valid = torch.zeros(T, device=device)
                
                # Sum the correct predictions vertically across all videos in the batch
                frame_by_frame_correct += ((predicted_class == target_label) & valid_mask).sum(dim=0)
                frame_by_frame_valid += valid_mask.sum(dim=0)
            
                # Track Start Accuracy
                start_mask = valid_mask[:, :T//4]
                total_start_correct += ((predicted_class[:, :T//4] == target_label[:, :T//4]) & start_mask).sum().item()
                total_start_samples += start_mask.sum().item()

                # Track End Accuracy
                end_mask = valid_mask[:, -T//4:]
                total_end_correct += ((predicted_class[:, -T//4:] == target_label[:, -T//4:]) & end_mask).sum().item()
                total_end_samples += end_mask.sum().item()
                
                # Track Final Frame Accuracy
                total_final_frame_correct += (predicted_class[:, -1] == target_label[:, -1]).sum().item()

                pbar.update(cfg.data.batch_size)
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        
        # Calculate final percentages
        val_accuracy = (correct_predictions / total_samples) * 100
        start_acc_pct = (total_start_correct / (total_start_samples + 1e-8)) * 100
        end_acc_pct = (total_end_correct / (total_end_samples + 1e-8)) * 100
        final_frame_accuracy = (total_final_frame_correct / len(val_dataset)) * 100

        print(f"\n--- Epoch {epoch + 1} Validation ---")
        print(f"Validation Loss: {avg_val_loss:.4f} | Overall Accuracy: {val_accuracy:.2f}%")
        print(f"Start Digit Acc: {start_acc_pct:.1f}% | End Digit Acc: {end_acc_pct:.1f}%")
        print(f"Absolute Final Frame Accuracy: {final_frame_accuracy:.2f}%\n")

        writer.add_scalar("validation/loss", avg_val_loss, global_step=global_step)
        writer.add_scalar("validation/accuracy", val_accuracy, global_step=global_step)
        writer.add_scalar("validation/start_digit_accuracy", start_acc_pct, global_step=global_step)
        writer.add_scalar("validation/end_digit_accuracy", end_acc_pct, global_step=global_step)
        writer.add_scalar("validation/final_frame_accuracy", final_frame_accuracy, global_step=global_step)

        if frame_by_frame_correct is not None:
            import matplotlib.pyplot as plt
            
            # Calculate percentage per frame: (Correct / Total Valid) * 100
            avg_timeline = (frame_by_frame_correct / frame_by_frame_valid.clamp(min=1)) * 100
            avg_timeline = avg_timeline.cpu().numpy()
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(avg_timeline, label="Average Accuracy", color="#1f77b4", linewidth=2.5)
            
            # Draw a vertical line exactly in the middle where the morph happens
            ax.axvline(x=len(avg_timeline)//2, color="red", linestyle="--", label="Morph Point")
            
            ax.set_title(f"Average Frame-by-Frame Accuracy (Epoch {epoch + 1})")
            ax.set_xlabel("Sub-sampled Frames")
            ax.set_ylabel("Accuracy (%)")
            ax.set_ylim(0, 105)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Send the figure to TensorBoard
            writer.add_figure("validation/average_timeline", fig, global_step=global_step)
            plt.close(fig) # Close the figure to free memory

        model.train()

if __name__ == "__main__":
    main()
