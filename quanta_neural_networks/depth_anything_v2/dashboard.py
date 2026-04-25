"""
Dashboard for Depth-Anything-v2 on a photon-cube
"""
import json
from math import ceil
from pathlib import Path

import hydra
import numpy as np
import streamlit as st
import torch
from einops import repeat, reduce
from hydra import initialize, compose
from jaxtyping import Float
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.pyplot import colormaps
from omegaconf import OmegaConf
from torch import Tensor

from quanta_neural_networks.depth_anything_v2.depth_anything import (
    DepthAnythingV2SSM,
    DepthAnythingV2,
)
from quanta_neural_networks.ops.array_ops import normalize
from quanta_neural_networks.ops.image import (
    intensity_from_empirical_mean,
)
from quanta_neural_networks.spad_data import (
    get_photon_cube,
    get_hot_pixel_mask,
)
from quanta_neural_networks.utils.dashboard import (
    save_image,
    save_video,
    setup_sidebar,
    setup_form,
    get_available_sub_config,
    _log_and_display,
)
from quanta_neural_networks.utils.plotting import color_depth
from quanta_neural_networks.utils.timer import CudaTimer, CPUTimer
from quanta_neural_networks.utils.train_utils import load_checkpoint

plt.rc("font", size=14)  # controls default text sizes


def _normalize_spatially(tensor: Float[Tensor, "h w t"]) -> Float[Tensor, "h w t"]:
    min_across_time = reduce(tensor, "h w t -> 1 1 t", "min")
    max_across_time = reduce(tensor, "h w t -> 1 1 t", "max")

    return (tensor - min_across_time) / (max_across_time - min_across_time)


def infer_depth(
    model: DepthAnythingV2SSM,
    photon_cube,
    hot_pixel_mask,
    method_kwargs,
    viz_kwargs,
    output_folder,
    label: str = "",
    spad_frame_rate: float = 96.8e3,
):
    Timer = CudaTimer if photon_cube.device.type == "cuda" else CPUTimer
    colors = colormaps["turbo"]

    device = photon_cube.device

    with Timer() as t_instance, st.spinner(text="Running depth model"), torch.autocast(
        device_type=device.type, dtype=torch.float16
    ):
        pred_depth_ll, t_index_ll = model.forward(
            photon_cube,
            bocpd_gamma=method_kwargs["bocpd_gamma"],
            min_window=method_kwargs["min_window"],
            online=True,
            quantile=viz_kwargs["quantile"],
            hot_pixel_mask=hot_pixel_mask,
        )

    logger.info(f"Depth reconstruction of shape {pred_depth_ll.shape}")

    pred_depth_ll = _normalize_spatially(pred_depth_ll)
    median_pred_depth = pred_depth_ll.median().item()

    save_video(
        color_depth(pred_depth_ll.cpu(), colors),
        output_folder,
        label=f"{label}_depth_video",
        downsample_factor=1,
        show=True,
    )

    metadata = {
        f"depther_{label}_runtime": float(t_instance),
    }
    _log_and_display(f"Runtime {t_instance:.3g} seconds.")

    del pred_depth_ll

    return metadata, median_pred_depth


@hydra.main(
    config_path=f"../../conf",
    config_name=f"{Path(__file__).parent.name}_{Path(__file__).stem}",
    version_base="1.2",
)
@torch.no_grad()
def main(cfg):
    # Get overrides from sidebar
    overrides_dict = {
        "scene": None,
        "data": lambda u: "blender_depth" in u,
        "model": lambda u: "depth_anything_v2" in u,
    }
    overrides_ll = []
    for sub_config, criteria in overrides_dict.items():
        override_file = get_available_sub_config(
            folder=sub_config,
            filter_criteria=criteria,
        )
        overrides_ll.append(f"{sub_config}={override_file}")

    st.title("Depth Anything v2 on Photons")

    hydra.core.global_hydra.GlobalHydra.instance().clear()

    colors = colormaps["turbo"]

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

            with st.spinner("Loading photon cube"):
                photon_cube_kwargs = {**cfg.scene, **scene_kwargs}
                photon_cube = get_photon_cube(**photon_cube_kwargs, dtype=np.float32)

                photon_cube = torch.from_numpy(photon_cube).to(device)

            # Hot pixel mask
            hot_pixel_mask = None
            if cfg.scene.get("hot_pixel"):
                hot_pixel_mask = get_hot_pixel_mask(
                    **cfg.scene.hot_pixel,
                    rotate_180=cfg.scene.get("rotate_180"),
                    flip_lr=cfg.scene.get("flip_lr"),
                    flip_ud=cfg.scene.get("flip_ud"),
                )

            # Load depth anything model
            depth_anything = DepthAnythingV2(**cfg.depth_anything.kwargs)
            depth_anything.eval()
            depth_anything = depth_anything.to(device)

            logger.info(
                f"Loading depth anything-v2 ckpt from {cfg.depth_anything.ckpt}"
            )
            load_checkpoint(
                depth_anything,
                Path(cfg.depth_anything.ckpt.folder),
                ckpt_file=cfg.depth_anything.ckpt.get("file"),
                ckpt_key="model",
            )

            # Load photon depther
            photon_depther = DepthAnythingV2SSM(**cfg.model.kwargs)
            # photon_depther.load_state_dict(depth_anything_ckpt, strict=False)
            load_checkpoint(
                photon_depther,
                Path(cfg.model.ckpt.folder),
                ckpt_file=cfg.model.ckpt.get("file"),
                ckpt_key="model",
            )
            photon_depther = photon_depther.to(device)
            photon_depther.eval()

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

            long_exposure_depth = None
            for tag, exposure_slice in zip(
                ["long_exposure", "short_exposure", "binary_frame"],
                [
                    np.s_[None:None],
                    np.s_[-viz_kwargs["short_exposure_frames"] :],
                    np.s_[-1:],
                ],
            ):
                title = tag.replace("_", " ").title()
                st.subheader(title)
                exposure = photon_cube[..., exposure_slice].mean(dim=-1)

                frame_depth = depth_anything.infer_image(
                    repeat(exposure.numpy(force=True) * 255, "h w -> h w c", c=3)
                )

                if tag == "long_exposure":
                    long_exposure_depth = frame_depth

                if viz_kwargs["invert_SPAD_response"]:
                    exposure = intensity_from_empirical_mean(
                        exposure, quantile=viz_kwargs["quantile"]
                    )

                save_image(
                    exposure,
                    output_folder,
                    hot_pixel_mask=hot_pixel_mask,
                    label=title,
                    show=True,
                    **viz_kwargs,
                )

                save_image(
                    color_depth(normalize(frame_depth), colors),
                    output_folder,
                    hot_pixel_mask=hot_pixel_mask,
                    label=f"Depth anything on {title.lower()}",
                    show=True,
                )

            st.subheader(f"Dense Inference")

            _metadata_update, median_pred_depth = infer_depth(
                photon_depther,
                photon_cube,
                hot_pixel_mask,
                method_kwargs,
                viz_kwargs,
                output_folder,
                spad_frame_rate=cfg.spad_frame_rate,
                label="dense",
            )

            metadata.update(_metadata_update)

            st.subheader("Normalized long_exposure depth")

            scale_factor = median_pred_depth / np.median(long_exposure_depth)
            long_exposure_depth = (long_exposure_depth * scale_factor * 1).clip(0, 1)

            save_image(
                color_depth(long_exposure_depth, colors),
                output_folder,
                hot_pixel_mask=hot_pixel_mask,
                label=f"Depth anything on {title.lower()}",
                show=True,
            )

            del frame_depth, photon_depther, photon_cube, depth_anything

            _log_and_display("Saving metadata")
            with open(output_folder / "depth_estimation_metadata.json", "w") as f:
                json.dump(metadata, f, indent=4, sort_keys=True)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
