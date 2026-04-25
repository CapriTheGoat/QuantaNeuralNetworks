"""
Dashboard for residual projection on a photon-cube
"""
import json
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
# noinspection PyUnresolvedReferences
import scienceplots
import streamlit as st
import torch
from hydra import initialize, compose
from loguru import logger
from omegaconf import OmegaConf

from quanta_neural_networks.reconstruction.efficient_ssd import EfficientSSD
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
from quanta_neural_networks.utils.timer import CudaTimer, CPUTimer
from quanta_neural_networks.utils.train_utils import load_checkpoint

plt.style.use("science")
# plt.rcParams["font.family"] = "sans-serif"
# plt.rcParams["font.sans-serif"] = ["Calibri"]


@hydra.main(
    config_path=f"../../conf",
    config_name=f"{Path(__file__).parent.name}_{Path(__file__).stem}",
    version_base="1.2",
)
@torch.no_grad()
def main(cfg):
    # Get overrides from sidebar
    overrides_dict = {"scene": None, "model": lambda u: "rdb" in u or "efficient" in u}
    overrides_ll = []
    for override_sub_config, filter_criteria in overrides_dict.items():
        override_file = get_available_sub_config(
            folder=override_sub_config, filter_criteria=filter_criteria
        )
        overrides_ll.append(f"{override_sub_config}={override_file}")

    st.title("Intensity Reconstruction from Photons")

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
            Timer = CudaTimer if device.type == "cuda" else CPUTimer

            st.info(f"Using device {device}")

            with st.spinner("Loading photon cube"):
                photon_cube_kwargs = {**cfg.scene, **scene_kwargs}
                _PHOTON_CUBE_KEYS = {"file", "initial_time_step", "num_time_step", "temporal_stride", "colorSPAD_col_correct", "colorSPAD_RGBW_CFA", "crop_sensor", "rotate_180", "flip_lr", "flip_ud"}
                photon_cube = get_photon_cube(**{k: v for k, v in photon_cube_kwargs.items() if k in _PHOTON_CUBE_KEYS}, dtype=torch.float32)
                h_orig, w_orig, t = photon_cube.shape

                photon_cube = photon_cube.to(device)

            # Hot pixel mask
            hot_pixel_mask = None
            if cfg.scene.get("hot_pixel"):
                hot_pixel_mask = get_hot_pixel_mask(
                    **cfg.scene.hot_pixel,
                    rotate_180=cfg.scene.get("rotate_180"),
                    flip_lr=cfg.scene.get("flip_lr"),
                    flip_ud=cfg.scene.get("flip_ud"),
                )

            cfg.model.kwargs.subsampling = method_kwargs["subsampling"]

            # Init model
            model = EfficientSSD(**cfg.model.kwargs)

            # Load ckpt
            load_checkpoint(
                model,
                Path(cfg.model.ckpt.folder),
                ckpt_file=cfg.model.ckpt.get("file"),
                ckpt_key="model",
            )
            model = model.to(device)
            model.eval()

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
                ["long_exposure", "short_exposure", "binary_frame"],
                [
                    photon_cube.mean(dim=-1),
                    photon_cube[..., -viz_kwargs["short_exposure_frames"] :].mean(
                        dim=-1
                    ),
                    photon_cube[..., -1],
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

            st.subheader(f"Reconstruction")
            _log_and_display("Dense eval")

            with Timer() as t_instance, st.spinner("Running dense eval"):
                out_ll, t_index_ll = model.forward_online(
                    photon_cube,
                    bocpd_gamma=method_kwargs["bocpd_gamma"],
                    hot_pixel_mask=hot_pixel_mask,
                )

            out_ll = out_ll.clamp(0, 1).cpu()[:h_orig, :w_orig]
            logger.info(f"Reconstruction of shape {out_ll.shape}")

            save_video(
                out_ll,
                output_folder,
                label="reconstruction",
                downsample_factor=1,
                show=True,
                **viz_kwargs,
            )

            metadata["dense_runtime"] = float(t_instance)
            _log_and_display(f"Runtime {t_instance:.3g} seconds.")
            logger.info("Done saving reconstruction")
            del out_ll

            del photon_cube

            with open(output_folder / "reconstruction_metadata.json", "w") as f:
                json.dump(metadata, f, indent=4, sort_keys=True)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
