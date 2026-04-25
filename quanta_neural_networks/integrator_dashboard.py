"""
Interactive dashboard for Bayesian integrator experimentation/visualization.
"""
from pathlib import Path

import hydra
import streamlit as st
import torch
from hydra import initialize, compose
from loguru import logger
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from omegaconf import OmegaConf

from quanta_neural_networks.integrator import PerPixelBayesian
from quanta_neural_networks.spad_data import get_photon_cube, get_hot_pixel_mask
from quanta_neural_networks.utils.dashboard import (
    save_video,
    setup_sidebar,
    setup_form,
    get_available_sub_config,
    save_image,
    _log_and_display,
)
from quanta_neural_networks.utils.timer import CudaTimer, CPUTimer


@hydra.main(
    config_path=f"../conf",
    config_name=Path(__file__).stem,
    version_base="1.2",
)
@torch.no_grad()
def main(cfg):
    """
    Launch an interactive dashboard for Bayesian integrator experiments.

    :param cfg: Configuration settings for dashboard run
    """
    # Get overrides from sidebar
    overrides_dict = {"scene": None}
    overrides_ll = []
    for override_sub_config in overrides_dict:
        override_file = get_available_sub_config(folder=override_sub_config)
        overrides_ll.append(f"{override_sub_config}={override_file}")

    st.title("Adaptive EMA on a Photon-Cube")

    hydra.core.global_hydra.GlobalHydra.instance().clear()

    device = (
        torch.device(cfg.device) if torch.cuda.is_available() else torch.device("cpu")
    )

    with initialize(
        config_path="../conf",
        version_base="1.2",
    ):
        cfg = compose(
            config_name=Path(__file__).stem,
            overrides=overrides_ll,
        )

        print(OmegaConf.to_yaml(cfg))
        scene_kwargs = setup_sidebar(cfg)

        simulate_button, method_kwargs, viz_kwargs = setup_form(cfg)

        if simulate_button:
            # Get photon cube
            with st.spinner("Loading photon cube"):
                photon_cube = get_photon_cube(**{**cfg.scene, **scene_kwargs}).to(
                    device
                )
                Timer = CudaTimer if device.type == "cuda" else CPUTimer

            # Hot pixel mask
            if cfg.scene.get("hot_pixel"):
                hot_pixel_mask = get_hot_pixel_mask(
                    **cfg.scene.hot_pixel,
                    rotate_180=cfg.scene.get("rotate_180"),
                    flip_lr=cfg.scene.get("flip_lr"),
                    flip_ud=cfg.scene.get("flip_ud"),
                )
            else:
                hot_pixel_mask = None

            # Display and save video
            output_folder = (
                Path(cfg.scene.name)
                / "adaptive_ema"
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

            integrator = PerPixelBayesian(
                bocpd_gamma=method_kwargs["bocpd_gamma"],
                memory_size=method_kwargs["memory_size"],
                subsampling=method_kwargs["subsampling"],
                hot_pixel_mask=hot_pixel_mask,
                min_filter_size=method_kwargs["min_filter_size"],
            )

            title = f"Adaptive EMA Reconstruction"
            st.subheader(title)

            with Timer() as t_instance, st.spinner(f"Running integrator"):
                recons_ll = integrator.process_photon_cube(photon_cube)

            fig, ax = plt.subplots()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)

            ax.set_title("Last frame window size")
            im = ax.imshow(
                integrator.sample_weight.numpy(force=True),
                cmap="cividis",
            )
            ax.axis("off")

            fig.colorbar(im, cax=cax, orientation="vertical")

            st.pyplot(fig)

            _log_and_display(f"{title} took {t_instance:.2g} seconds.")
            save_video(
                recons_ll,
                output_folder,
                label="adaptive_ema",
                show=True,
                **viz_kwargs,
                downsample_factor=1,
            )

            del photon_cube

            del recons_ll
            if device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
