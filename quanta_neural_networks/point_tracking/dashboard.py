"""
Dashboard for point tracking on a photon cube
"""
import json
from math import ceil
from pathlib import Path

import hydra
import numpy as np
import streamlit as st
import torch
from einops import repeat
from hydra import initialize, compose
from jaxtyping import Float
from loguru import logger
from omegaconf import OmegaConf
from torch import Tensor

from quanta_neural_networks.ops.image import (
    intensity_from_empirical_mean,
)
from quanta_neural_networks.point_tracking.pips import PointTracker
from quanta_neural_networks.point_tracking.utils import Visualizer
from quanta_neural_networks.reconstruction.efficient_ssd import EfficientSSD
from quanta_neural_networks.spad_data import (
    get_photon_cube,
    get_hot_pixel_mask,
)
from quanta_neural_networks.utils.dashboard import (
    save_image,
    setup_sidebar,
    setup_form,
    get_available_sub_config,
    _log_and_display,
)
from quanta_neural_networks.utils.timer import CudaTimer, CPUTimer
from quanta_neural_networks.utils.train_utils import load_checkpoint


def infer_tracks(
    model: PointTracker,
    photon_cube,
    hot_pixel_mask,
    recons_ll,
    recons_t_index_ll,
    method_kwargs,
    viz_kwargs,
    output_folder,
    model_name: str,
    label: str = "",
    spad_frame_rate: float = 96.8e3,
):
    visualizer = Visualizer(save_dir=output_folder, fps=viz_kwargs["output_fps"])

    h, w, t = photon_cube.shape
    device = photon_cube.device

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(16, h - 16, steps=viz_kwargs["points_per_side"]),
        torch.linspace(16, w - 16, steps=viz_kwargs["points_per_side"]),
    )
    coords_init = torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2).to(device)

    Timer = CudaTimer if photon_cube.device.type == "cuda" else CPUTimer

    with Timer() as t_instance, st.spinner(text=f"Running {model_name}"):
        (
            coord_predictions_ll,
            feature_map_ll,
            t_index_ll,
        ) = model.forward(
            photon_cube,
            coords_init,
            bocpd_gamma=method_kwargs["bocpd_gamma"],
            online=True,
            chaining_length=16,
            hot_pixel_mask=hot_pixel_mask,
        )

    pred_trajectory: Float[
        Tensor, "num_frame num_points num_coords"
    ] = coord_predictions_ll[-1]

    num_frame, num_points, _ = pred_trajectory.shape

    # Suppress minimal displacements
    pred_displacements = pred_trajectory[1:] - pred_trajectory[:-1]
    masked_displacements = (
        pred_displacements.sum(dim=0).norm(dim=-1, p=2)
        < viz_kwargs["displacement_suppression_threshold"]
    )

    pred_trajectory[:, masked_displacements] = repeat(
        coords_init,
        "num_points num_coords -> num_frame num_points num_coords",
        num_frame=num_frame,
    )[:, masked_displacements]

    pred_trajectory = pred_trajectory

    recons_t_index_ll = np.array(recons_t_index_ll)
    mask = np.zeros_like(recons_t_index_ll, dtype=bool)
    for t_index in t_index_ll:
        mask = mask | (recons_t_index_ll == t_index)

    visualizer.visualize(
        recons_ll[..., mask],
        pred_trajectory=pred_trajectory,
        filename=label,
    )
    st.video(str(output_folder / f"{label}.mp4"))

    return t_instance


@hydra.main(
    config_path=f"../../conf",
    config_name=f"{Path(__file__).parent.name}_{Path(__file__).stem}",
    version_base="1.2",
)
@torch.no_grad()
def main(cfg):
    overrides_dict = {
        "scene": None,
        "model": lambda u: "pips" in u,
    }
    overrides_ll = []
    for sub_config, criteria in overrides_dict.items():
        override_file = get_available_sub_config(
            folder=sub_config,
            filter_criteria=criteria,
        )
        overrides_ll.append(f"{sub_config}={override_file}")

    st.title("Particle Tracking on Photons")
    hydra.core.global_hydra.GlobalHydra.instance().clear()

    with initialize(
        config_path="../../conf",
        version_base="1.2",
    ):
        cfg = compose(
            config_name=f"{Path(__file__).parent.name}_{Path(__file__).stem}",
            overrides=overrides_ll,
        )

        print(OmegaConf.to_yaml(cfg))
        scene_kwargs = setup_sidebar(cfg)

        simulate_button, method_kwargs, viz_kwargs = setup_form(
            cfg, label="Learned Projection Parameters"
        )

        if simulate_button:
            device = (
                torch.device(viz_kwargs["device"])
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            st.info(f"Using device {device}")

            cfg.model.kwargs.subsampling = method_kwargs["subsampling"]

            # Get photon cube
            subsampling = cfg.model.kwargs.subsampling
            scene_kwargs["num_time_step"] = (
                ceil(scene_kwargs["num_time_step"] / subsampling) * subsampling
            )

            logger.info(f"Instantiating {cfg.model.name}")
            model = PointTracker(**cfg.model.kwargs)

            logger.info(f"Loading Pips feature extractor from {cfg.model.pips_ckpt}")
            pips_ckpt = torch.load(cfg.model.pips_ckpt, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(
                pips_ckpt, strict=False
            )

            logger.info(f"Missing keys {missing_keys}")
            logger.info(f"Unexpected keys {unexpected_keys}")

            load_checkpoint(
                model,
                Path(cfg.model.ckpt.folder),
                ckpt_file=cfg.model.ckpt.get("file"),
                ckpt_key="model",
                strict=False,
            )
            model = model.to(device)
            model.eval()

            with st.spinner("Loading photon cube"):
                photon_cube_kwargs = {**cfg.scene, **scene_kwargs}
                photon_cube = get_photon_cube(**photon_cube_kwargs, dtype=np.float32)
                h_orig, w_orig, t = photon_cube.shape

            photon_cube = photon_cube[
                : int(h_orig // model.spatial_stride) * model.spatial_stride,
                : int(w_orig // model.spatial_stride) * model.spatial_stride,
            ]

            height, width, _ = photon_cube.shape

            photon_cube = torch.from_numpy(photon_cube).to(device)

            # Hot pixel mask
            hot_pixel_mask = None
            if cfg.scene.get("hot_pixel"):
                hot_pixel_mask = get_hot_pixel_mask(
                    **cfg.scene.hot_pixel,
                    rotate_180=cfg.scene.get("rotate_180"),
                    flip_lr=cfg.scene.get("flip_lr"),
                    flip_ud=cfg.scene.get("flip_ud"),
                )[:height, :width]
                logger.info(f"Done inpainting photon cube")

            # Metadata
            metadata = {
                **method_kwargs,
                **viz_kwargs,
                **scene_kwargs,
            }

            # Display and save video
            output_folder = (
                Path(cfg.scene.name)
                / cfg.model.name
                / f"{range(scene_kwargs['initial_time_step'], scene_kwargs['initial_time_step'] + scene_kwargs['num_time_step'], scene_kwargs['temporal_stride'])}"
            )
            output_folder.mkdir(exist_ok=True, parents=True)

            logger.info(f"Photon cube of shape {photon_cube.shape}")

            for exposure_name, exposure in zip(
                ["binary_frame", "long_exposure", "short_exposure"],
                [
                    photon_cube[..., -1],
                    photon_cube.mean(dim=-1),
                    photon_cube[..., -viz_kwargs["short_exposure_frames"] :].mean(
                        dim=-1
                    ),
                ],
            ):
                st.subheader(exposure_name.replace("_", " ").title())
                save_image(
                    exposure,
                    output_folder,
                    label=exposure_name,
                    hot_pixel_mask=hot_pixel_mask,
                    **viz_kwargs,
                )

            logger.info("Running sequentially")
            st.subheader("Dense Inference")

            with st.spinner("Running intensity restoration for overlay"):
                # Init model
                recons_model = EfficientSSD(**cfg.recons_model.kwargs)
                # Load ckpt
                load_checkpoint(
                    recons_model,
                    Path(cfg.recons_model.ckpt.folder),
                    ckpt_file=cfg.recons_model.ckpt.get("file"),
                    ckpt_key="model",
                )
                recons_model = recons_model.to(device)
                recons_model.eval()

                recons_ll, recons_t_index_ll = recons_model.forward_online(
                    photon_cube,
                    bocpd_gamma=method_kwargs["bocpd_gamma"],
                    hot_pixel_mask=hot_pixel_mask,
                )

                if viz_kwargs["invert_SPAD_response"]:
                    recons_ll = intensity_from_empirical_mean(
                        recons_ll, quantile=viz_kwargs["quantile"]
                    )

            t_instance = infer_tracks(
                model,
                photon_cube,
                hot_pixel_mask,
                recons_ll,
                recons_t_index_ll,
                method_kwargs,
                viz_kwargs,
                output_folder,
                model_name=cfg.model.name,
                label="dense_tracking",
                spad_frame_rate=cfg.spad_frame_rate,
            )

            metadata[f"{cfg.model.name}_dense_runtime"] = float(t_instance)
            _log_and_display(f"Runtime {t_instance:.3g} seconds.")
            logger.info("Done saving point-tracking frames")

            _log_and_display("Saving metadata")
            with open(output_folder / "point_tracking_metadata.json", "w") as f:
                json.dump(metadata, f, indent=4, sort_keys=True)

            del photon_cube, recons_ll

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
