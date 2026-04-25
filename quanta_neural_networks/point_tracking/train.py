"""
Main training script for point_tracking on photon-cube Kubric dataset.
"""
from pathlib import Path

import hydra
import numpy as np
import torch
from einops import rearrange
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from quanta_neural_networks.ssd import SSD
from quanta_neural_networks.ops.array_ops import (
    reduce_masked_mean,
    loguniform,
    normalize,
)
from quanta_neural_networks.point_tracking.dataloader import TrackingDataset
from quanta_neural_networks.point_tracking.loss import (
    sequence_loss,
)
from quanta_neural_networks.point_tracking.pips import PointTracker
from quanta_neural_networks.point_tracking.utils import Visualizer
from quanta_neural_networks.utils.feature_vis import cast_features_to_rgb
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
    Train a point tracking model using the provided configuration.

    :param cfg: Hydra-configured settings object
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Taichi init
    logger.info(f"Using device {device}")

    # model
    model = PointTracker(**cfg.model.kwargs)
    logger.info(f"Loading Pips feature extractor from {cfg.model.pips_ckpt}")
    pips_ckpt = torch.load(cfg.model.pips_ckpt, map_location="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(pips_ckpt, strict=False)

    logger.info(f"Missing keys {missing_keys}")
    logger.info(f"Unexpected keys {unexpected_keys}")

    # Load onto device
    model = model.to(device)
    for module in model.modules():
        if isinstance(module, SSD):
            module.parallel_mode = cfg.model.get("parallel_mode", False)

    train_dataset = TrackingDataset(
        **cfg.data.train, trajectory_subsampling=model.fnet.subsampling
    )
    val_dataset = TrackingDataset(
        **cfg.data.val, trajectory_subsampling=model.fnet.subsampling
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset,
        shuffle=True,
        batch_size=1,
        num_workers=0,
    )

    # Setup optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), **cfg.optim)
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
    visualizer = Visualizer()

    epoch_start, global_step = resume_or_finetune(
        model, optimizer, cfg.model.ckpt, scheduler
    )

    epoch_start = global_step // len(train_dataset)

    for epoch in range(epoch_start, cfg.num_epoch):
        logger.info(f"Train epoch {epoch + 1} | Global step {global_step}")
        model.train()

        with tqdm(total=len(train_dataset), dynamic_ncols=True) as pbar:
            for index, batch in enumerate(train_dataloader):
                sequence_name, intensity_ll, gt_trajectory, unoccluded = batch
                intensity_ll = intensity_ll.squeeze(0).to(device)
                gt_trajectory = gt_trajectory.squeeze(0).to(device)
                unoccluded = unoccluded.float().squeeze(0).to(device)

                # Simulate photon cube
                max_probability = loguniform(
                    cfg.photon_cube.min_probability, cfg.photon_cube.max_probability
                )
                photon_cube = simulate_photon_cube(
                    intensity_ll, max_probability=max_probability
                )

                bocpd_gamma = loguniform(cfg.bocpd_gamma.min, cfg.bocpd_gamma.max)

                (
                    coord_predictions_ll,
                    feature_map_ll,
                    t_index_ll,
                ) = model.forward(
                    photon_cube,
                    gt_trajectory[0],
                    bocpd_gamma=bocpd_gamma,
                    t_init=None if cfg.data.use_half_gt else 0,
                )

                loss = sequence_loss(
                    gt_trajectory,
                    coord_predictions_ll,
                    unoccluded,
                    **cfg.loss.sequence_loss,
                )
                loss.backward()

                pred_trajectory = coord_predictions_ll[-1]
                l1_distance = reduce_masked_mean(
                    (pred_trajectory - gt_trajectory).abs().sum(dim=-1), unoccluded
                )

                if index % cfg.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                # Logging
                global_step += 1
                pbar.update(1)

                if index % cfg.logging.scalar_interval == 0:
                    pbar.set_description(
                        f"Train epoch {epoch + 1} | Total loss {loss.item():.4f}"
                    )

                    writer.add_scalar(
                        "training/loss", loss.item(), global_step=global_step
                    )

                    writer.add_scalar(
                        "training/l1_dist", l1_distance.item(), global_step=global_step
                    )

                    writer.add_scalar(
                        "learning_rate",
                        scheduler.get_last_lr()[0],  # is a list
                        global_step=global_step,
                    )

                if index % cfg.logging.image_interval == 0:
                    # log images
                    writer.add_images(
                        "training/long_exposure",
                        normalize(photon_cube.mean(dim=-1).detach()),
                        dataformats="HW",
                        global_step=global_step,
                    )
                    writer.add_images(
                        "training/photon_slice",
                        photon_cube[..., -1].detach(),
                        dataformats="HW",
                        global_step=global_step,
                    )
                    intensity_stride = intensity_ll.shape[-1] // gt_trajectory.shape[0]
                    if cfg.data.use_half_gt:
                        intensity_stride = intensity_stride // 2

                    intensity_strided_ll = intensity_ll[
                        ..., intensity_ll.shape[-1] // 2 :: intensity_stride
                    ]
                    writer.add_images(
                        "training/intensity_images",
                        rearrange(
                            intensity_strided_ll.detach(),
                            "h w t -> t 1 h w",
                        ),
                        dataformats="NCHW",
                        global_step=global_step,
                    )

                    writer.add_images(
                        "training/feature_map",
                        cast_features_to_rgb(feature_map_ll.detach())[0],
                        dataformats="NCHW",
                        global_step=global_step,
                    )

                    viz_slice = np.s_[:, : cfg.logging.max_viz_points]

                    visualizer.visualize(
                        photon_cube[..., photon_cube.shape[-1] // 2 :],
                        pred_trajectory=pred_trajectory[viz_slice],
                        gt_trajectory=gt_trajectory[viz_slice],
                        visibility=unoccluded[viz_slice],
                        writer=writer,
                        step=global_step,
                        tag="training/tracks",
                    )

                    visualizer.visualize(
                        intensity_ll[..., photon_cube.shape[-1] // 2 :],
                        pred_trajectory=pred_trajectory[viz_slice],
                        visibility=unoccluded[viz_slice],
                        writer=writer,
                        step=global_step,
                        tag="training/tracks_pred_only",
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
        average_loss = 0.0
        average_l1_distance = 0.0

        with tqdm(
            total=len(val_dataset), dynamic_ncols=True
        ) as pbar, torch.inference_mode():
            for index, batch in enumerate(val_dataloader):
                sequence_name, intensity_ll, gt_trajectory, unoccluded = batch
                intensity_ll = intensity_ll.squeeze(0).to(device)
                gt_trajectory = gt_trajectory.squeeze(0).to(device)
                unoccluded = unoccluded.float().squeeze(0).to(device)

                # Simulate photon cube
                max_probability = loguniform(
                    cfg.photon_cube.min_probability, cfg.photon_cube.max_probability
                )
                photon_cube = simulate_photon_cube(
                    intensity_ll, max_probability=max_probability
                )

                bocpd_gamma = loguniform(cfg.bocpd_gamma.min, cfg.bocpd_gamma.max)

                (
                    coord_predictions_ll,
                    feature_map_ll,
                    t_index_ll,
                ) = model.forward(
                    photon_cube,
                    gt_trajectory[0],
                    bocpd_gamma=bocpd_gamma,
                    t_init=None if cfg.data.use_half_gt else 0,
                    online=True,
                )

                loss = sequence_loss(
                    gt_trajectory,
                    coord_predictions_ll,
                    unoccluded,
                    **cfg.loss.sequence_loss,
                )
                average_loss += (loss.item() - average_loss) / (index + 1)

                pred_trajectory = coord_predictions_ll[-1]
                l1_distance = reduce_masked_mean(
                    (pred_trajectory - gt_trajectory).abs().sum(dim=-1), unoccluded
                )
                average_l1_distance += (l1_distance.item() - average_l1_distance) / (
                    index + 1
                )

                pbar.update(1)

            # Logging
            writer.add_scalar("validation/loss", average_loss, global_step=global_step)
            writer.add_scalar(
                "validation/l1_dist", average_l1_distance, global_step=global_step
            )

            log_str = " | ".join(
                [
                    f"Val epoch {epoch + 1}",
                    f"Average loss {average_loss:.4f}",
                ]
            )
            pbar.set_description(log_str)
            logger.info(log_str)

            # log images
            writer.add_images(
                "validation/long_exposure",
                normalize(photon_cube.mean(dim=-1).detach()),
                dataformats="HW",
                global_step=global_step,
            )
            writer.add_images(
                "validation/photon_slice",
                photon_cube[..., -1].detach(),
                dataformats="HW",
                global_step=global_step,
            )
            intensity_stride = intensity_ll.shape[-1] // gt_trajectory.shape[0]
            if cfg.data.use_half_gt:
                intensity_stride = intensity_stride // 2

            intensity_strided_ll = intensity_ll[
                ..., intensity_ll.shape[-1] // 2 :: intensity_stride
            ]
            writer.add_images(
                "validation/intensity_images",
                rearrange(
                    intensity_strided_ll.detach(),
                    "h w t -> t 1 h w",
                ),
                dataformats="NCHW",
                global_step=global_step,
            )

            writer.add_images(
                "validation/feature_map",
                cast_features_to_rgb(feature_map_ll.detach())[0],
                dataformats="NCHW",
                global_step=global_step,
            )

            viz_slice = np.s_[:, : cfg.logging.max_viz_points]

            visualizer.visualize(
                photon_cube[..., photon_cube.shape[-1] // 2 :],
                pred_trajectory=pred_trajectory[viz_slice],
                gt_trajectory=gt_trajectory[viz_slice],
                visibility=unoccluded[viz_slice],
                writer=writer,
                step=global_step,
                tag="validation/tracks",
            )

            visualizer.visualize(
                intensity_ll[..., photon_cube.shape[-1] // 2 :],
                pred_trajectory=pred_trajectory[viz_slice],
                visibility=unoccluded[viz_slice],
                writer=writer,
                step=global_step,
                tag="validation/tracks_pred_only",
            )


if __name__ == "__main__":
    main()
