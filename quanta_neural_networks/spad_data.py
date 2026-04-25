"""
SPAD (Single-Photon Avalanche Diode) dataset utilities.
Includes functions for interpolation, correction, and loading data cubes.
"""
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from loguru import logger
from torch import Tensor


def interpolate_W(photon_cube: Tensor, cfa_mask: Tensor) -> Tensor:
    """
    Interpolate the photon cube over the color filter array (CFA).

    :param photon_cube: Input photon cube tensor
    :param cfa_mask: Color filter array mask
    :return: Interpolated photon cube
    """
    if not isinstance(photon_cube, Tensor):
        photon_cube = torch.from_numpy(photon_cube)
    if not isinstance(cfa_mask, Tensor):
        cfa_mask = torch.from_numpy(cfa_mask)

    device = photon_cube.device
    h, w, t = photon_cube.shape
    cfa_mask = cfa_mask.to(device)

    # We use a grouped convolution to process all time steps in parallel.
    # The number of groups must equal the number of input channels (t).
    groups = t

    # Base kernel for a single channel
    base_kernel = torch.tensor(
        [[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float32, device=device
    )

    # Repeat the kernel for each group. Shape becomes [out_channels, in_channels_per_group, H, W]
    # which is [t, 1, 3, 3]
    kernel = base_kernel.unsqueeze(0).unsqueeze(0).repeat(groups, 1, 1, 1)

    cube_ch_first = photon_cube.permute(2, 0, 1).unsqueeze(0).float()
    padding = (kernel.shape[-1] - 1) // 2

    # Add groups=groups to the conv2d call
    neighbor_sum = F.conv2d(cube_ch_first, kernel, padding=padding, groups=groups)
    neighbor_count = F.conv2d(
        torch.ones_like(cube_ch_first), kernel, padding=padding, groups=groups
    )

    prob = (neighbor_sum / neighbor_count.clamp(min=1e-9)).squeeze(0).permute(1, 2, 0)
    new_values = torch.rand_like(prob) < prob
    interpolated_cube = torch.where(cfa_mask.unsqueeze(-1), new_values, photon_cube)

    return interpolated_cube


def interpolate_hot_pixel(photon_cube: Tensor, hot_pixel_mask: Tensor) -> Tensor:
    """
    Inpaint or correct hot pixels in a photon cube based on mask locations.

    :param photon_cube: Input photon cube
    :param hot_pixel_mask: Mask of hot pixel locations
    :return: Hot-pixel-corrected photon cube
    """
    if not isinstance(photon_cube, Tensor):
        photon_cube = torch.from_numpy(photon_cube)
    if not isinstance(hot_pixel_mask, Tensor):
        hot_pixel_mask = torch.from_numpy(hot_pixel_mask)

    device = photon_cube.device
    h, w, t = photon_cube.shape
    hot_pixel_mask = hot_pixel_mask.to(device)

    groups = t

    # Base kernel for a single channel (sums 8 neighbors)
    base_kernel = torch.tensor(
        [[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=torch.float32, device=device
    )

    # Repeat the kernel for each group. Shape -> [t, 1, 3, 3]
    kernel = base_kernel.unsqueeze(0).unsqueeze(0).repeat(groups, 1, 1, 1)

    cube_ch_first = photon_cube.permute(2, 0, 1).unsqueeze(0).float()
    padding = (kernel.shape[-1] - 1) // 2

    # Add groups=groups to the conv2d call
    neighbor_sum = F.conv2d(cube_ch_first, kernel, padding=padding, groups=groups)
    neighbor_count = F.conv2d(
        torch.ones_like(cube_ch_first), kernel, padding=padding, groups=groups
    )

    prob = (neighbor_sum / neighbor_count.clamp(min=1e-9)).squeeze(0).permute(1, 2, 0)
    new_values = torch.rand_like(prob) < prob
    interpolated_cube = torch.where(
        hot_pixel_mask.unsqueeze(-1), new_values, photon_cube
    )

    return interpolated_cube


def colorSPAD_correct_func(photon_cube: np.ndarray, colorSPAD_RGBW_CFA: str) -> Tensor:
    """
    Apply SPAD color channel correction using RGBW color filter array.

    :param photon_cube: Raw SPAD measurement array
    :param colorSPAD_RGBW_CFA: CFA type or configuration string
    :return: Corrected photon cube tensor
    """
    photon_cube_tensor = torch.from_numpy(photon_cube)

    # Perform column swaps and cropping with tensor slicing
    photon_cube_cropped = torch.zeros(
        (254, 496, photon_cube_tensor.shape[-1]),
        dtype=photon_cube_tensor.dtype,
        device=photon_cube_tensor.device,
    )
    photon_cube_cropped[:254, :496] = photon_cube_tensor[2:, :496]

    # Atomically swap columns using a temporary variable
    col1 = photon_cube_tensor[2:, 252:256].clone()
    col2 = photon_cube_tensor[2:, 260:264].clone()
    photon_cube_cropped[:254, 252:256] = col2
    photon_cube_cropped[:254, 260:264] = col1

    photon_cube_tensor = photon_cube_cropped

    if colorSPAD_RGBW_CFA:
        cfa = cv2.imread(str(colorSPAD_RGBW_CFA))[2:, :496, ::-1]  # BGR -> RGB
        cfa_tensor = torch.from_numpy(cfa.copy()).to(photon_cube_tensor.device)
        mask = cfa_tensor.float().mean(dim=-1) < 255
        photon_cube_tensor = interpolate_W(photon_cube_tensor.to(torch.bool), mask)

    return photon_cube_tensor


def get_photon_cube(
    file: str, initial_time_step: int = 0, num_time_step: int = 1000,
    temporal_stride: int = 1, colorSPAD_col_correct: bool = False,
    colorSPAD_RGBW_CFA: str = None, dtype=torch.float32,
    memmap_cube=None, crop_sensor: bool = False,
    rotate_180: bool = False, flip_lr: bool = False, flip_ud: bool = False,
) -> Tensor:
    """
    Load photon cube data from file with options for color correction and stride.

    :param file: Path to data file
    :param initial_time_step: Initial timestep offset
    :param num_time_step: Number of time steps to load
    :param temporal_stride: Stride over time axis
    :param colorSPAD_col_correct: Apply color SPAD correction
    :param colorSPAD_RGBW_CFA: Color filter array (CFA) configuration
    :param dtype: Output tensor dtype
    :return: Loaded photon cube tensor
    """
    if isinstance(file, str):
        file = Path(file)

    final_time_step = initial_time_step + num_time_step * temporal_stride
    logger.info(
        f"Loading photon cube from {file}, Time steps {initial_time_step}:{final_time_step}:{temporal_stride}"
    )

    if memmap_cube is None:
        memmap_cube = np.load(file, mmap_mode="r")

    # Slicing and unpacking is fast in numpy, convert to tensor after
    photon_cube_np = memmap_cube[initial_time_step:final_time_step:temporal_stride]
    photon_cube_np = rearrange(np.unpackbits(photon_cube_np, axis=-1), "t h w -> h w t")

    # Convert to PyTorch Tensor
    photon_cube = torch.from_numpy(photon_cube_np.copy())

    if colorSPAD_col_correct:
        photon_cube = colorSPAD_correct_func(
            photon_cube.numpy(), str(colorSPAD_RGBW_CFA)
        )

    if crop_sensor:
        photon_cube = photon_cube[2:, :496].clone()

    if rotate_180:
        photon_cube = torch.flip(photon_cube, dims=[0, 1])

    if flip_lr:
        photon_cube = torch.flip(photon_cube, dims=[1])

    if flip_ud:
        photon_cube = torch.flip(photon_cube, dims=[0])

    # Remove top and bottom rows
    photon_cube = photon_cube[1:-1]

    return photon_cube.to(dtype)


def get_hot_pixel_mask(
    mask_path: Path,
    rotate_180: bool = False,
    flip_lr: bool = False,
    flip_ud: bool = False,
    **kwargs,
) -> Tensor:
    """
    Get hot_pixel_mask from a .npy file, returning a PyTorch Tensor.
    """
    hot_pixel_mask_np = np.load(mask_path).astype(np.uint8)
    hot_pixel_mask = torch.from_numpy(hot_pixel_mask_np)

    if rotate_180:
        hot_pixel_mask = torch.flip(hot_pixel_mask, dims=[0, 1])

    if flip_lr:
        hot_pixel_mask = torch.flip(hot_pixel_mask, dims=[1])

    if flip_ud:
        hot_pixel_mask = torch.flip(hot_pixel_mask, dims=[0])

    # Remove top and bottom rows
    hot_pixel_mask = hot_pixel_mask[1:-1]

    return hot_pixel_mask
