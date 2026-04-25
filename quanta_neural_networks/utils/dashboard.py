"""
Dashboard helpers for Streamlit app and experiment interface, including video/image save, plotting, and sidebar/forms.
"""
import subprocess
from pathlib import Path
from typing import Union, List, Callable

import cv2
import numpy as np
import streamlit as st
import torch
from einops import repeat, rearrange, reduce
from hydra.utils import get_original_cwd
from jaxtyping import Float, Bool
from loguru import logger
from matplotlib import pyplot as plt, animation
from millify import millify
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import ndarray
from omegaconf import OmegaConf
from scipy.ndimage import gaussian_filter1d
from streamlit_image_coordinates import streamlit_image_coordinates
from torch import Tensor

from quanta_neural_networks.ops.array_ops import float_to_uint8
from quanta_neural_networks.ops.image import (
    nearest_neighbor_inpaint,
    intensity_from_empirical_mean,
)


def save_image(
    image: Union[Float[Tensor, "h w"], Float[ndarray, "h w"]],
    output_folder: Path | str,
    label: str,
    hot_pixel_mask: Bool[np.ndarray, "h w"] = None,
    invert_SPAD_response: bool = False,
    quantile: float = 0.975,
    show: bool = True,
    select_pixel: bool = False,
    **kwargs,
):
    """
    Save or display an image with possible inpainting and quantile normalization.
    :param image: Image tensor or array
    :param output_folder: Where to save .png
    :param label: Caption or filename
    :param hot_pixel_mask: Optional binary mask
    :param invert_SPAD_response: If True, invert empirical mean to intensity
    :param quantile: Quantile for normalization
    :param show: Display in Streamlit
    :param select_pixel: Enable pixel selection UI
    :return: Value if select_pixel is True, else None
    """
    if isinstance(hot_pixel_mask, np.ndarray):
        image = nearest_neighbor_inpaint(image, hot_pixel_mask)

    if isinstance(image, Tensor):
        image = image.numpy(force=True)

    if invert_SPAD_response:
        image = intensity_from_empirical_mean(image, quantile=quantile)
    image = float_to_uint8(image)

    if output_folder:
        output_folder.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(
            str(output_folder / f"{label.replace(' ', '_').lower()}.png"),
            image[..., ::-1] if image.ndim == 3 else image,
        )

    if select_pixel:
        value = streamlit_image_coordinates(
            repeat(image, "h w -> h w c", c=3),
            key="numpy",
        )
        st.caption(label.title())
        return value

    if show:
        st.image(image, caption=label.title(), output_format="PNG")


def save_video(
    recons_cube: Union[Float[Tensor, "h w t"], Float[np.ndarray, "h w t"]],
    output_folder: Path,
    label: str,
    hot_pixel_mask: Bool[np.ndarray, "h w"] = None,
    downsample_factor: int = 10,
    invert_SPAD_response: bool = False,
    quantile: float = 0.975,
    output_fps: int = 30,
    show: bool = True,
    **kwargs,
):
    """
    Save a stack/sequence of frames as an h264/MP4 video using ffmpeg piping.
    :param recons_cube: Stack to save/display
    :param output_folder: Where to save .mp4
    :param label: Filename
    :param hot_pixel_mask: Inpaint if given
    :param downsample_factor: Skip every n-th frame
    :param invert_SPAD_response: Apply physical inverse to SPAD output
    :param quantile: For normalization
    :param output_fps: Frames per second
    :param show: Show with Streamlit.video
    :return: None
    """
    if show:
        logger.info(f"Saving {label.title()}")

    if isinstance(recons_cube, np.ndarray):
        recons_cube = torch.from_numpy(recons_cube)

    if isinstance(hot_pixel_mask, np.ndarray):
        recons_cube = nearest_neighbor_inpaint(recons_cube, hot_pixel_mask)

    if invert_SPAD_response:
        recons_cube = intensity_from_empirical_mean(recons_cube, quantile=quantile)

    recons_cube = recons_cube[:, :, ::downsample_factor].cpu()
    if recons_cube.ndim == 3:
        recons_video = repeat(recons_cube, "h w t -> t h w c", c=3)
    elif recons_cube.ndim == 4:
        recons_video = rearrange(recons_cube, "h w t c -> t h w c")[..., :3]

    video_path = output_folder / f"{label}.mp4"
    output_folder.mkdir(exist_ok=True, parents=True)

    # 1. Prepare the video data as a uint8 NumPy array
    video_data = (recons_video.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    num_frames, height, width, _ = video_data.shape

    # 2. Construct the FFmpeg command
    command = [
        "ffmpeg",
        "-y",  # Overwrite output file if it exists
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",  # Frame size
        "-pix_fmt",
        "rgb24",  # Input pixel format
        "-r",
        str(output_fps),  # Frames per second
        "-i",
        "-",  # The input comes from a pipe
        "-c:v",
        "libx264",  # Codec for the output
        "-pix_fmt",
        "yuv420p",  # Pixel format for broad compatibility
        "-crf",
        "23",  # Constant Rate Factor (quality level, 23 is a good default)
        str(video_path),
    ]

    # 3. Open a subprocess pipe and write the frames
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )

        # Write all frames' raw byte data to the process's stdin
        process.stdin.write(video_data.tobytes())

        # Close stdin and wait for the process to finish
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            logger.error(f"FFmpeg failed with return code {process.returncode}")
            # Decode stderr from bytes to string for printing
            logger.error(f"FFmpeg stderr: {stderr.decode('utf-8')}")

    except FileNotFoundError:
        logger.error(
            "FFmpeg not found. Please ensure it is installed and in your system's PATH."
        )
        return

    if show:
        st.video(str(video_path))


def get_available_sub_config(folder: str = "scene", filter_criteria: Callable = None) -> List[str]:
    """
    List available configuration names from a folder, optionally filtering.
    """
    # Get available scene configs
    cfg_ll = Path(f"{get_original_cwd()}/conf/{folder}").glob("*.yaml")
    cfg_ll = [file.stem for file in sorted(list(cfg_ll))]

    if filter_criteria is not None:
        cfg_ll = [cfg for cfg in cfg_ll if filter_criteria(cfg)]

    cfg = st.sidebar.selectbox(f"{folder.title()} Config", cfg_ll)
    return cfg


def setup_sidebar(cfg):
    """
    Build sidebar numeric/input widgets to configure settings interactively.
    :param cfg: page configuration object
    :return: Dict of user selections/inputs
    """
    return {
        "initial_time_step": st.sidebar.number_input(
            "Initial Time Step",
            min_value=0,
            max_value=10_000_000,
            value=cfg.scene.initial_time_step,
            step=10,
        ),
        "num_time_step": st.sidebar.number_input(
            "Num Time Step",
            min_value=-1,
            max_value=200_000,
            value=cfg.scene.num_time_step,
            step=1,
        ),
        "temporal_stride": st.sidebar.number_input(
            "Temporal stride",
            min_value=1,
            max_value=100,
            value=1,
            step=1,
            help="striding in the time axis. "
            "Alter this to 'speed' up the photon-cube.",
        ),
        "chunk_size": st.sidebar.number_input(
            "Chunk size",
            min_value=1,
            max_value=20_000,
            value=8192,
            step=1,
            help="Chunk size to read photon cube in.",
        ),
    }


def setup_form(cfg, label: str = "Method Form", button_name: str = "Run", **submit_kwargs):
    """
    Build and display a Streamlit form interface for method and visualization params.
    :param cfg: Configuration
    :param label: Form label
    :param button_name: Button label
    :param submit_kwargs: Extra kwargs for submit button
    :return: (form_submit_return, ...) tuple of kwargs from form inputs
    """
    # Setup HiPPO form
    form = st.form(label)
    with form:
        col_1, col_2 = st.columns(2)
        kwargs_ll = []

        for name, col in zip(
            ["method_params", "viz"],
            [col_1, col_2],
        ):
            with col:
                st.subheader(name.replace("_", " ").title())

                params = OmegaConf.to_container(cfg.get(name), resolve=True)
                kwargs_ll.append(
                    {
                        k: getattr(st, v["input_type"])(
                            k.replace("_", " ").title(), **v["input_kwargs"]
                        )
                        for k, v in params.items()
                    }
                )

        simulate_button = st.form_submit_button(button_name, **submit_kwargs)
    return simulate_button, *kwargs_ll


def _log_and_display(text):
    """
    Print text and log via Streamlit/loguru.
    """
    st.text(str(text))
    logger.info(text)


def plot_layer_stats(
    stats_dict: dict,
    output_folder: Path,
    photon_cube_shape: tuple[int, int, int] | torch.Size,
    spad_frame_rate: float = 96.8e3,
    downsample_factor: int = 8,
    tag: str = "dense",
):
    """
    Build interactive plots and save animated videos showing network activations/churn/statistics.
    :param stats_dict: Per-layer statistics dict
    :param output_folder: Video save location
    :param photon_cube_shape: Shape for normalization/bitrates
    :param spad_frame_rate: SPAD frame rate for time axis
    :param downsample_factor: Steps for animation
    :param tag: Label for filenames and UI
    """
    def _frame_update(index):
        im_1.set_data(activations_ll[index])
        im_2.set_data(changes_ll[index])
        tx_1.set_text(f"Layer name {layer_name} | Time index {t_index_ll[index]}")

        overall_density = changes_ll[index].mean() * 100
        pixel_density = (changes_ll[index] > 0).float().mean() * 100
        tx_2.set_text(
            f"Overall density {overall_density: .2g}% | pixel density {pixel_density: .2g}%"
        )

        ax_3.plot(t_index_ll[: index + 1], temporal_changes[: index + 1])
        return im_1, im_2, tx_1, tx_2

    with st.expander("Activation stats"):
        for layer_name in stats_dict:
            t_index_ll = stats_dict[layer_name]["t_index_ll"]
            activations_ll = stats_dict[layer_name]["pca"]
            changes_ll = stats_dict[layer_name]["changes"]

            bits = sum(stats_dict[layer_name]["bits"])

            fig, (ax_1, ax_2, ax_3) = plt.subplots(ncols=3, figsize=(18, 5))

            st.text(layer_name)

            if isinstance(activations_ll, list):
                activations_ll = torch.stack(activations_ll, dim=0)
            if isinstance(changes_ll, list):
                changes_ll = torch.stack(changes_ll, dim=0)

            activations_ll = activations_ll.cpu()
            activations_ll = rearrange(activations_ll, "n c h w -> n h w c")

            changes_ll = changes_ll.cpu()
            temporal_changes = reduce(changes_ll, "n h w -> n", "mean")

            # Downsample
            down_slice = np.s_[::downsample_factor]
            activations_ll = activations_ll[down_slice]
            changes_ll = changes_ll[down_slice]
            temporal_changes = temporal_changes[down_slice]

            im_1 = ax_1.imshow(activations_ll[0], cmap="gray")
            im_1.set_clim(0, 1.0)
            ax_1.set_title("First 3 PCA")

            im_2 = ax_2.imshow(changes_ll[0], cmap="cividis")
            im_2.set_clim(changes_ll.min(), changes_ll.max())
            ax_2.set_title("Thresholded differences")

            tx_1 = ax_1.set_xlabel(
                f"Layer name {layer_name} | Time index {t_index_ll[0]}"
            )

            overall_density = changes_ll[0].mean() * 100
            pixel_density = (changes_ll[0] > 0).float().mean() * 100
            tx_2 = ax_2.set_xlabel(
                f"Overall density {overall_density: .2g}% | pixel density {pixel_density: .2g}%"
            )

            ax_3.set_title("Changes across time")
            ax_3.plot(t_index_ll[0], temporal_changes[0])
            ax_3.set_xlabel(f"Time index")

            # create an axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            divider = make_axes_locatable(ax_2)
            cax = divider.append_axes("right", size="5%", pad=0.05)

            height, width, t = photon_cube_shape
            bits_per_pixel = bits / (height * width)
            bits_per_pixel_per_second = bits_per_pixel * spad_frame_rate / t
            fig.suptitle(
                f"{millify(bits, precision=2)} bits | "
                f"{millify(bits_per_pixel, precision=2)} bits/pixel | "
                f"{millify(bits_per_pixel_per_second, precision=2)} bits/pixel/second"
            )

            fig.colorbar(im_2, cax=cax)
            plt.tight_layout()

            ani = animation.FuncAnimation(
                fig=fig,
                func=_frame_update,
                frames=len(changes_ll),
                interval=30,
            )

            ssm_changes_path = output_folder / f"{tag}_{layer_name}_activations.mp4"

            ani.save(
                filename=str(ssm_changes_path),
                writer=animation.FFMpegWriter(fps=60, bitrate=-1, codec="h264"),
                dpi=200,
            )

            st.video(str(ssm_changes_path))


def plot_effective_framerate(
    t_index_ll: list[int],
    output_folder: Path,
    spad_frame_rate: float = 96.8e3,
    t_warmup: int = None,
    sigma: int = 7,
    figsize: tuple[float, float] = (7.83, 7),
    show: bool = True,
) -> plt.Figure:
    """
    Plot, save, and (optionally) display effective frame rate as a function of time.
    """
    fig, ax_ll = plt.subplots(figsize=figsize, nrows=2, sharex=False)

    gap_ll = np.diff(t_index_ll, prepend=0)
    gap_ll = gaussian_filter1d(gap_ll, sigma)
    frame_rate_ll = spad_frame_rate / gap_ll

    ax_ll[0].semilogy(frame_rate_ll, "k", linewidth=2)

    # yticks = [2e1, 2e2, 2e3, 2e4]

    # Clip the range as much as we can
    yticks = np.logspace(
        np.log2(frame_rate_ll.min()),
        np.log2(frame_rate_ll.max()),
        num=4,
        dtype=int,
        base=2,
    )

    ax_ll[1].semilogy(t_index_ll, frame_rate_ll, "|-k", linewidth=2)
    ax_ll[1].vlines(
        x=t_index_ll,
        ymin=yticks.min() / 2,
        ymax=frame_rate_ll,
        color="Grey",
        linestyles="--",
        linewidth=0.5,
    )
    ax_ll[0].set_xlabel("Output (frame) index")
    ax_ll[1].set_xlabel("Quanta frame index")

    t_index_ll = np.array(t_index_ll)
    title = f"Total samples {len(t_index_ll)}"
    if t_warmup:
        title += f" | Post warmup {(t_index_ll > t_warmup).sum()} samples"
    fig.suptitle(title)

    for e in range(2):
        ax_ll[e].set_yticks(yticks, yticks)
        ax_ll[e].set_ylim(yticks.min() / 2, yticks.max() * 2)

        ax_ll[e].set_ylabel("Frame rate (Hz)")
        ax_ll[e].grid()

    fig.tight_layout()
    plt.savefig(
        output_folder / "effective_framerate.pdf",
        dpi=150,
        transparent=True,
    )
    if show:
        st.pyplot(fig)
    return fig
