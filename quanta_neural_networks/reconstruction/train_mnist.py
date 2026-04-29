"""
Training entrypoint for reconstruction models.
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
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from quanta_neural_networks.ssd import SSD
from quanta_neural_networks.ops.array_ops import loguniform
from quanta_neural_networks.ops.metrics import PSNR
from quanta_neural_networks.reconstruction.dataloader import IntensityCubeSimulatedNPY
from quanta_neural_networks.reconstruction.efficient_ssd import EfficientSSD
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
def main(cfg):
    """
    Train a reconstruction model using provided configuration.

    :param cfg: Configuration/parameters (e.g. Hydra config)
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    logger.info(f"Using device {device}")

    train_dataset = IntensityCubeSimulatedNPY(**cfg.data.train)
    val_dataset = IntensityCubeSimulatedNPY(**cfg.data.val)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=1,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        prefetch_factor=2,
    )
    val_dataloader = DataLoader(
        val_dataset, shuffle=True, batch_size=1, num_workers=cfg.data.num_workers
    )

    # Init model
    model = EfficientSSD(**cfg.model.kwargs).to(device)

    for module in model.modules():
        if isinstance(module, SSD):
            module.parallel_mode = cfg.model.get("parallel_mode", False)

    # Setup optimizers, lr scheduler
    optimizer = torch.optim.AdamW(params=model.parameters(), **cfg.optim)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.num_epoch * len(train_dataloader) // cfg.gradient_accumulation_steps,
        **cfg.scheduler,
    )

    ckpt_dir = Path(cfg.model.ckpt.folder)
    ckpt_dir.mkdir(exist_ok=True, parents=True)

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

    psnr = PSNR()

    for epoch in range(epoch_start, cfg.num_epoch):
        logger.info(f"Train epoch {epoch + 1} | Global step {global_step}")

        with tqdm(total=len(train_dataset), dynamic_ncols=True) as pbar:
            model.train()
            for index, batch in enumerate(train_dataloader):
                video_name, photon_cube, intensity_ll = batch

                photon_cube = photon_cube.squeeze(0).to(device)
                intensity_ll = intensity_ll.squeeze(0).to(device)

                bocpd_gamma = loguniform(cfg.bocpd_gamma.min, cfg.bocpd_gamma.max)

                output_ll, t_index_ll = model.forward(
                    photon_cube, bocpd_gamma=bocpd_gamma
                )

                _, _, output_t = output_ll.shape

                if cfg.data.use_half_gt:
                    output_ll = output_ll[..., output_t // 2 :]
                    t_index_ll = t_index_ll[output_t // 2 :]

                with torch.no_grad():
                    intensity_ll = intensity_ll[..., np.array(t_index_ll) - 1]

                loss = getattr(F, cfg.loss.name)(output_ll, intensity_ll)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)

                if (index + 1) % cfg.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                # Logging
                global_step += cfg.data.batch_size
                pbar.update(cfg.data.batch_size)

                if index % cfg.logging.scalar_interval == 0:
                    pbar.set_description(
                        f"Train epoch {epoch + 1} | loss {loss.item():.3f}"
                    )

                    writer.add_scalar(
                        "training/loss", loss.item(), global_step=global_step
                    )

                    writer.add_scalar(
                        "learning_rate",
                        scheduler.get_last_lr()[0],  # is a list
                        global_step=global_step,
                    )

                if index % cfg.logging.image_interval == 0:
                    # log images
                    writer.add_images(
                        f"training/long_exposure",
                        rearrange(photon_cube.mean(dim=-1).detach(), "h w -> 1 1 h w"),
                        dataformats="NCHW",
                        global_step=global_step,
                    )
                    writer.add_images(
                        f"training/photon_slice",
                        rearrange(photon_cube[..., -1].detach(), "h w -> 1 1 h w"),
                        dataformats="NCHW",
                        global_step=global_step,
                    )
                    writer.add_images(
                        f"training/target",
                        rearrange(intensity_ll.detach(), "h w t -> t 1 h w")[
                            -cfg.logging.num_images :
                        ],
                        dataformats="NCHW",
                        global_step=global_step,
                    )
                    writer.add_images(
                        "training/output",
                        rearrange(output_ll.clamp(0, 1).detach(), "h w t -> t 1 h w")[
                            -cfg.logging.num_images :
                        ],
                        dataformats="NCHW",
                        global_step=global_step,
                    )

        # Save checkpoints
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
        logger.info(f"Val epoch {epoch + 1}")
        average_psnr = 0.0
        average_ssim = 0.0

        with tqdm(
            total=len(val_dataset), dynamic_ncols=True
        ) as pbar, torch.inference_mode():
            for index, batch in enumerate(val_dataloader):
                video_name, intensity_ll = batch
                intensity_ll = intensity_ll.squeeze(0).to(device)

                # Simulate photon cube
                max_probability = loguniform(
                    cfg.photon_cube.min_probability, cfg.photon_cube.max_probability
                )
                intensity_ll *= max_probability
                photon_cube = simulate_photon_cube(intensity_ll)

                bocpd_gamma = loguniform(cfg.bocpd_gamma.min, cfg.bocpd_gamma.max)

                output_ll, t_index_ll = model.forward_online(
                    photon_cube, bocpd_gamma=bocpd_gamma
                )

                output_ll = output_ll.clamp(0, 1)

                _, _, output_t = output_ll.shape

                if cfg.data.use_half_gt:
                    output_ll = output_ll[..., output_t // 2 :]
                    t_index_ll = t_index_ll[output_t // 2 :]

                intensity_ll = intensity_ll[..., np.array(t_index_ll) - 1]
                batch_psnr = psnr(output_ll, intensity_ll)
                batch_ssim = ssim(
                    rearrange(output_ll, "h w t -> t 1 h w"),
                    rearrange(intensity_ll, "h w t -> t 1 h w"),
                    data_range=1.0,
                )

                average_psnr += (batch_psnr.mean().item() - average_psnr) / (index + 1)
                average_ssim += (batch_ssim.mean().item() - average_ssim) / (index + 1)

                pbar.update(1)

            # Logging
            writer.add_scalar(
                "validation/psnr",
                average_psnr,
                global_step=global_step,
            )
            writer.add_scalar("validation/ssim", average_ssim, global_step=global_step)
            log_str = " | ".join(
                [
                    f"Val epoch {epoch + 1}",
                    f"PSNR {average_psnr:.2f}",
                    f"SSIM {average_ssim:.3f}",
                ]
            )
            pbar.set_description(log_str)
            logger.info(log_str)

            # log images
            writer.add_images(
                f"validation/long_exposure",
                rearrange(photon_cube.mean(dim=-1).detach(), "h w -> 1 1 h w"),
                dataformats="NCHW",
                global_step=global_step,
            )
            writer.add_images(
                f"validation/photon_slice",
                rearrange(photon_cube[..., -1].detach(), "h w -> 1 1 h w"),
                dataformats="NCHW",
                global_step=global_step,
            )
            writer.add_images(
                f"validation/target",
                rearrange(intensity_ll, "h w t -> t 1 h w")[
                    -cfg.logging.num_images :
                ].detach(),
                dataformats="NCHW",
                global_step=global_step,
            )
            writer.add_images(
                "validation/output",
                rearrange(output_ll.clamp(0, 1), "h w t -> t 1 h w")[
                    -cfg.logging.num_images :
                ].detach(),
                dataformats="NCHW",
                global_step=global_step,
            )


if __name__ == "__main__":
    main()
