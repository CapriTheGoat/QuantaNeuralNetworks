"""
Finetune depth-anything-v2 converted to a photon-cube network using SSDs
"""
from pathlib import Path

import hydra
import numpy as np
import torch
from einops import rearrange
from loguru import logger
from matplotlib.pyplot import colormaps
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from quanta_neural_networks.depth_anything_v2.dataloader import BlenderDepth, XVFIDepth
from quanta_neural_networks.depth_anything_v2.depth_anything import (
    DepthAnythingV2SSM,
    DepthAnythingV2,
)
from quanta_neural_networks.depth_anything_v2.loss import MAELoss, GradientLoss
from quanta_neural_networks.ssd import SSD
from quanta_neural_networks.ops.array_ops import normalize, loguniform
from quanta_neural_networks.utils.hydra import print_and_save_cfg
from quanta_neural_networks.utils.plotting import color_depth
from quanta_neural_networks.utils.train_utils import (
    resume_or_finetune,
    simulate_photon_cube,
    unfreeze_module,
)

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

    # Load depth anything model
    depth_anything = DepthAnythingV2(**cfg.depth_anything.kwargs)
    depth_anything.eval()
    depth_anything = depth_anything.to(device)

    logger.info(f"Loading depth anything-v2 ckpt from {cfg.depth_anything.ckpt}")
    depth_anything_ckpt = torch.load(
        cfg.depth_anything.ckpt, map_location="cpu", weights_only=False
    )
    depth_anything.load_state_dict(depth_anything_ckpt)

    photon_depth_anything = DepthAnythingV2SSM(**cfg.model.kwargs)
    photon_depth_anything = photon_depth_anything.to(device)
    for module in photon_depth_anything.modules():
        if isinstance(module, SSD):
            module.parallel_mode = cfg.model.get("parallel_mode", False)
    # Unfreeze depth head
    if not cfg.model.freeze_depth_head:
        logger.info("Thawing DPT head.")
        unfreeze_module(photon_depth_anything.depth_head)
    photon_depth_anything.load_state_dict(depth_anything_ckpt, strict=False)

    depth_anything_pixel_mean = rearrange(
        photon_depth_anything.pretrained.mean, "c -> 1 c 1 1"
    )
    depth_anything_pixel_std = rearrange(
        photon_depth_anything.pretrained.std, "c -> 1 c 1 1"
    )

    depth_stride = photon_depth_anything.pretrained.subsampling // (
        cfg.data.max_intensity_frames // cfg.data.max_depth_frames
    )
    logger.info(f"Depth stride {depth_stride}")
    train_dataset = BlenderDepth(
        **{**cfg.data.train, "reshape_size": cfg.depth_anything.kwargs.image_size},
        depth_stride=depth_stride,
    )
    val_dataset = BlenderDepth(
        **{**cfg.data.val, "reshape_size": cfg.depth_anything.kwargs.image_size},
        depth_stride=depth_stride,
    )

    min_depth, max_depth = cfg.data.min_depth, cfg.data.max_depth
    logger.info(f"Min depth {min_depth}m max depth {max_depth}m")

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )
    val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=1, num_workers=0)

    # Setup optimizer
    optimizer = torch.optim.Adam(params=photon_depth_anything.parameters(), **cfg.optim)
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
        photon_depth_anything, optimizer, cfg.model.ckpt, scheduler
    )
    epoch_start = global_step // len(train_dataset)

    colors = colormaps["turbo"]

    #  Loss terms
    mae_loss = MAELoss(
        **cfg.loss.mae_loss,
        on_depth=False,
        min_depth=1e-2,
        max_depth=10,
    )
    gradient_loss = GradientLoss(
        **cfg.loss.gradient_loss,
        on_depth=False,
        min_depth=1e-2,
        max_depth=10,
    )

    for epoch in range(epoch_start, cfg.num_epoch):
        logger.info(f"Train epoch {epoch + 1} | Global step {global_step}")

        with tqdm(total=len(train_dataset), dynamic_ncols=True) as pbar:
            photon_depth_anything.train()
            for index, batch in enumerate(train_dataloader):
                sequence_name, intensity_ll, gt_depth_ll = batch
                intensity_ll = intensity_ll.squeeze(0).to(device)
                gt_depth_ll = gt_depth_ll.squeeze(0).to(device)
                gt_disparity_ll = 1 / gt_depth_ll.clamp(1 / max_depth, 1 / min_depth)

                # Simulate photon cube
                max_probability = loguniform(
                    cfg.photon_cube.min_probability, cfg.photon_cube.max_probability
                )
                photon_cube = simulate_photon_cube(
                    intensity_ll, max_probability=max_probability
                )

                bocpd_gamma = loguniform(cfg.bocpd_gamma.min, cfg.bocpd_gamma.max)
                min_window = loguniform(cfg.min_window.min, cfg.min_window.max)

                pred_depth_ll, t_index_ll = photon_depth_anything.forward(
                    photon_cube, bocpd_gamma=bocpd_gamma, min_window=min_window
                )

                t_index_ll = np.array(t_index_ll)
                if cfg.data.use_half_gt:
                    # Use only latter half for loss
                    half_slice = np.s_[pred_depth_ll.shape[-1] // 2 :]
                    pred_depth_ll = pred_depth_ll[..., pred_depth_ll.shape[-1] // 2 :]
                    t_index_ll = t_index_ll[half_slice]

                with torch.no_grad():
                    # select intensity frames for dino_depth, reshape
                    depth_anything_intensity_ll = intensity_ll[..., t_index_ll - 1]
                    depth_anything_intensity_ll = rearrange(
                        depth_anything_intensity_ll, "h w t -> t 1 h w"
                    )

                    # Normalize
                    depth_anything_intensity_normalized_ll = (
                        depth_anything_intensity_ll - depth_anything_pixel_mean
                    ) / depth_anything_pixel_std

                    pred_depth_anything_ll = depth_anything(
                        depth_anything_intensity_normalized_ll
                    )
                    pred_depth_anything_ll = rearrange(
                        pred_depth_anything_ll, "t h w -> h w t"
                    )

                    # Naive depth
                    long_exposure = rearrange(
                        photon_cube.mean(dim=-1), "h w -> 1 1 h w"
                    )
                    long_exposure_normalized_ll = (
                        long_exposure - depth_anything_pixel_mean
                    ) / depth_anything_pixel_std

                    long_exposure_depth_anything = depth_anything(
                        long_exposure_normalized_ll
                    )
                    long_exposure_depth_anything = rearrange(
                        long_exposure_depth_anything, "1 h w -> h w 1"
                    )

                    # Short exposure
                    short_exposure = rearrange(
                        photon_cube[..., -cfg.short_exposure_frames :].mean(dim=-1),
                        "h w -> 1 1 h w",
                    )
                    short_exposure_normalized_ll = (
                        short_exposure - depth_anything_pixel_mean
                    ) / depth_anything_pixel_std

                    short_exposure_depth_anything = depth_anything(
                        short_exposure_normalized_ll
                    )
                    short_exposure_depth_anything = rearrange(
                        short_exposure_depth_anything, "1 h w -> h w 1"
                    )

                mae_loss_score = mae_loss(pred_depth_ll, gt_disparity_ll)
                gradient_loss_score = gradient_loss(pred_depth_ll, gt_disparity_ll)

                loss = mae_loss_score + gradient_loss_score
                loss.backward()

                if index % cfg.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                # Logging
                global_step += 1
                pbar.update(1)

                if index % cfg.logging.scalar_interval == 0:
                    pbar.set_description(
                        f"Train epoch {epoch + 1} | MAE + grad loss {loss.item():.4f}"
                    )

                    writer.add_scalar(
                        "training/loss", loss.item(), global_step=global_step
                    )

                    writer.add_scalar(
                        "learning_rate",
                        scheduler.get_last_lr()[0],  # is a list
                        global_step=global_step,
                    )
                    writer.add_scalar(
                        "training/mae_loss",
                        mae_loss_score.item(),
                        global_step=global_step,
                    )
                    writer.add_scalar(
                        "training/gradient_loss",
                        gradient_loss_score.item(),
                        global_step=global_step,
                    )

                if index % cfg.logging.image_interval == 0:
                    # log images
                    writer.add_images(
                        f"training/long_exposure",
                        normalize(long_exposure.squeeze().detach()),
                        dataformats="HW",
                        global_step=global_step,
                    )
                    writer.add_images(
                        f"training/short_exposure",
                        normalize(short_exposure.squeeze().detach()),
                        dataformats="HW",
                        global_step=global_step,
                    )
                    writer.add_images(
                        f"training/photon_slice",
                        photon_cube[..., -1].detach(),
                        dataformats="HW",
                        global_step=global_step,
                    )
                    writer.add_images(
                        f"training/intensity_ll",
                        rearrange(
                            intensity_ll[..., t_index_ll - 1], "h w t -> t 1 h w"
                        ),
                        dataformats="NCHW",
                        global_step=global_step,
                    )
                    with torch.no_grad():
                        for label, depth in zip(
                            [
                                "pred_ours",
                                "pred_depth_anything",
                                "pred_depth_anything_long_exposure",
                                "pred_depth_anything_short_exposure",
                                "gt_depth",
                            ],
                            [
                                pred_depth_ll,
                                pred_depth_anything_ll,
                                long_exposure_depth_anything,
                                short_exposure_depth_anything,
                                gt_disparity_ll,
                            ],
                        ):
                            depth_colored = color_depth(
                                normalize(depth).detach().cpu(),
                                colors,
                            )

                            depth_colored = rearrange(
                                depth_colored, "h w t c -> t c h w"
                            )

                            writer.add_images(
                                f"training/{label}",
                                depth_colored,
                                dataformats="NCHW",
                                global_step=global_step,
                            )

        # Save checkpoints
        if epoch % cfg.model.ckpt.epoch_interval == 0:
            logger.info(f"Saving state to {ckpt_dir}")

            torch.save(
                {
                    "model": photon_depth_anything.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "global_step": global_step,
                },
                ckpt_dir / "checkpoint.pth",
            )

        # Validation
        photon_depth_anything.eval()
        logger.info(f"Val epoch {epoch + 1}")
        average_loss = 0.0

        with tqdm(
            total=len(val_dataset), dynamic_ncols=True
        ) as pbar, torch.inference_mode():
            for index, batch in enumerate(val_dataloader):
                sequence_name, intensity_ll, gt_depth_ll = batch
                intensity_ll = intensity_ll.squeeze(0).to(device)
                gt_depth_ll = gt_depth_ll.squeeze(0).to(device)
                gt_disparity_ll = 1 / gt_depth_ll.clamp(1 / max_depth, 1 / min_depth)

                # Simulate photon cube
                max_probability = loguniform(
                    cfg.photon_cube.min_probability, cfg.photon_cube.max_probability
                )
                photon_cube = simulate_photon_cube(
                    intensity_ll, max_probability=max_probability
                )

                bocpd_gamma = loguniform(cfg.bocpd_gamma.min, cfg.bocpd_gamma.max)
                min_window = loguniform(cfg.min_window.min, cfg.min_window.max)

                pred_depth_ll, t_index_ll = photon_depth_anything.forward(
                    photon_cube,
                    bocpd_gamma=bocpd_gamma,
                    min_window=min_window,
                    online=True,
                )

                t_index_ll = np.array(t_index_ll)
                if cfg.data.use_half_gt:
                    # Use only latter half for loss
                    half_slice = np.s_[pred_depth_ll.shape[-1] // 2 :]
                    pred_depth_ll = pred_depth_ll[..., pred_depth_ll.shape[-1] // 2 :]
                    t_index_ll = t_index_ll[half_slice]

                # select intensity frames for dino_depth, reshape
                depth_anything_intensity_ll = intensity_ll[..., t_index_ll - 1]
                depth_anything_intensity_ll = rearrange(
                    depth_anything_intensity_ll, "h w t -> t 1 h w"
                )

                # Normalize
                depth_anything_intensity_normalized_ll = (
                    depth_anything_intensity_ll - depth_anything_pixel_mean
                ) / depth_anything_pixel_std

                pred_depth_anything_ll = depth_anything(
                    depth_anything_intensity_normalized_ll
                )
                pred_depth_anything_ll = rearrange(
                    pred_depth_anything_ll, "t h w -> h w t"
                )

                # Naive depth
                long_exposure = rearrange(photon_cube.mean(dim=-1), "h w -> 1 1 h w")
                long_exposure_normalized_ll = (
                    long_exposure - depth_anything_pixel_mean
                ) / depth_anything_pixel_std

                long_exposure_depth_anything = depth_anything(
                    long_exposure_normalized_ll
                )
                long_exposure_depth_anything = rearrange(
                    long_exposure_depth_anything, "1 h w -> h w 1"
                )

                # Short exposure
                short_exposure = rearrange(
                    photon_cube[..., -cfg.short_exposure_frames :].mean(dim=-1),
                    "h w -> 1 1 h w",
                )
                short_exposure_normalized_ll = (
                    short_exposure - depth_anything_pixel_mean
                ) / depth_anything_pixel_std

                short_exposure_depth_anything = depth_anything(
                    short_exposure_normalized_ll
                )
                short_exposure_depth_anything = rearrange(
                    short_exposure_depth_anything, "1 h w -> h w 1"
                )

                mae_loss_score = mae_loss(pred_depth_ll, gt_disparity_ll)
                gradient_loss_score = gradient_loss(pred_depth_ll, gt_disparity_ll)

                loss = mae_loss_score + gradient_loss_score

                average_loss += (loss.item() - average_loss) / (index + 1)

                pbar.update(1)

            # Logging
            writer.add_scalar(
                "validation/loss",
                average_loss,
                global_step=global_step,
            )
            log_str = " | ".join(
                [
                    f"Val epoch {epoch + 1}",
                    f"MAE + grad loss {average_loss:.4f}",
                ]
            )
            pbar.set_description(log_str)
            logger.info(log_str)

            # log images
            writer.add_images(
                f"validation/long_exposure",
                normalize(long_exposure.squeeze().detach()),
                dataformats="HW",
                global_step=global_step,
            )
            writer.add_images(
                f"validation/short_exposure",
                normalize(short_exposure.squeeze().detach()),
                dataformats="HW",
                global_step=global_step,
            )
            writer.add_images(
                f"validation/photon_slice",
                photon_cube[..., -1].detach(),
                dataformats="HW",
                global_step=global_step,
            )
            writer.add_images(
                f"validation/intensity_ll",
                rearrange(intensity_ll[..., t_index_ll - 1], "h w t -> t 1 h w"),
                dataformats="NCHW",
                global_step=global_step,
            )
            writer.add_scalar(
                "validation/mae_loss",
                mae_loss_score.item(),
                global_step=global_step,
            )
            writer.add_scalar(
                "validation/gradient_loss",
                gradient_loss_score.item(),
                global_step=global_step,
            )
            for label, depth in zip(
                [
                    "pred_ours",
                    "pred_depth_anything",
                    "pred_depth_anything_long_exposure",
                    "pred_depth_anything_short_exposure",
                    "gt_depth",
                ],
                [
                    pred_depth_ll,
                    pred_depth_anything_ll,
                    long_exposure_depth_anything,
                    short_exposure_depth_anything,
                    gt_disparity_ll,
                ],
            ):
                depth_colored = color_depth(
                    normalize(depth).detach().cpu(),
                    colors,
                )

                depth_colored = rearrange(depth_colored, "h w t c -> t c h w")

                writer.add_images(
                    f"validation/{label}",
                    depth_colored,
                    dataformats="NCHW",
                    global_step=global_step,
                )


if __name__ == "__main__":
    main()
