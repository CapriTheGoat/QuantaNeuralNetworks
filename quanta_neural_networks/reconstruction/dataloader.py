"""
Data loading utilities for reconstruction tasks.

This module provides data loading functionality for reconstruction tasks,
including image and video file detection, intensity cube extraction,
and dataset classes for reconstruction training.
"""

import imghdr
from math import ceil
from pathlib import Path
from random import randint
from typing import Tuple

import cv2
import numpy as np
import torch
import torchvision
from einops import rearrange, repeat
from jaxtyping import Float
from natsort import natsorted
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.io import VideoReader
from torchvision.transforms import v2
from tqdm import tqdm

from quanta_neural_networks.utils.train_utils import as_tuple


def is_image(file_path: str | Path) -> bool:
    """
    Check if a file is an image using imghdr.
    
    :param file_path: The path to the file
    :return: True if the file is an image, False otherwise
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        return False
    try:
        return imghdr.what(file_path) is not None
    except FileNotFoundError:
        return False


def is_video_file(file_path: str | Path) -> bool:
    """
    Check if a file is a video based on its extension.
    
    :param file_path: The path to the file
    :return: True if the file is a video, False otherwise
    """
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
    path = Path(file_path)
    return path.is_file() and path.suffix.lower() in video_extensions


def get_intensity_cube(
    path: Path,
    max_frames: int = None,
    reshape_size: int | tuple[int, int] = None,
    crop_size: int | tuple[int, int] = None,
    stride: int = 1,
) -> Float[Tensor, "height width time"]:
    """
    Extract intensity cube from video file.
    
    Returns intensity cube with values in [0...1]. Reshapes first, then crops.
    
    :param path: Video file path
    :param max_frames: Number of frames to read
    :param reshape_size: Reshape size for the video
    :param crop_size: Crop size for the video
    :param stride: Stride for frame sampling
    :return: Intensity cube tensor of shape (height, width, time)
    """

    if reshape_size is not None:
        reshape_size = as_tuple(reshape_size, size=2)
    if crop_size is not None:
        crop_size = as_tuple(crop_size, size=2)

    if is_video_file(path):
        reader = VideoReader(str(path), "video")
    elif path.is_dir():
        reader = natsorted(list(path.glob("*")))
        # reader = [file for file in reader if is_image(file)]
        # if len(reader) == 0:
        #     raise AssertionError(f"No valid image files found at {path}")
    else:
        raise NotImplementedError

    intensity_ll = []
    for e, file in enumerate(reader):
        if (e + 1) % stride == 0:
            # C, H, W
            if is_video_file(path):
                intensity = file["data"].float()
            else:
                intensity: np.ndarray = rearrange(
                    cv2.imread(str(file), -1)[..., ::-1], "h w c -> c h w"
                )
                intensity: Tensor = torch.from_numpy(intensity.copy())

            intensity_ll.append(intensity)

        if max_frames and e + 1 == max_frames:
            break

    # T, C, H , W
    intensity_ll = torch.stack(intensity_ll, dim=0)
    intensity_ll = intensity_ll.mean(dim=1, keepdim=True)

    if reshape_size:
        intensity_ll = v2.functional.resize(
            intensity_ll,
            size=reshape_size,
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
        )

    if crop_size:
        intensity_ll = intensity_ll[:, :, : crop_size[0], : crop_size[1]]

    intensity_ll = rearrange(intensity_ll, "t 1 h w -> h w t") / 255.0
    return intensity_ll


class IntensityCube(Dataset):
    def __init__(
        self,
        intensity_location: str | list[str, ...],
        reshape_size: Tuple[int, int] = None,
        crop_size: Tuple[int, int] = None,
        intensity_fps: int = 16_000,
        photon_cube_fps: int = 64_000,
        max_time_step: int = 2048,
    ):
        """
        :param intensity_location: The top-level directory containing the dataset
        """

        if isinstance(intensity_location, str):
            intensity_location = [intensity_location]

        # Collect all video files in all directories listed
        path_ll = []
        for video_dir in intensity_location:
            path_ll += list(Path(video_dir).glob("*.mp4"))

        # Load information about each video in the dataset.
        self.path_ll = natsorted(path_ll)

        if crop_size is not None:
            crop_size = as_tuple(crop_size, size=2)
        self.crop_size = crop_size
        self.reshape_size = reshape_size

        self.max_time_step = max_time_step
        self.photon_cube_oversampling = ceil(photon_cube_fps / intensity_fps)
        self._max_intensity_frames = ceil(max_time_step / self.photon_cube_oversampling)

    def __len__(self):
        return len(self.path_ll)

    def __getitem__(self, index):
        path = self.path_ll[index]
        video_name = path.name
        intensity_ll = get_intensity_cube(
            path, self._max_intensity_frames, self.reshape_size
        )

        intensity_ll = repeat(
            intensity_ll,
            "h w t -> h w (t num_repeat)",
            num_repeat=self.photon_cube_oversampling,
        )

        if self.crop_size:
            h, w, _ = intensity_ll.shape
            crop_i = randint(0, h - self.crop_size[0])
            crop_j = randint(0, w - self.crop_size[1])
            crop_slice = np.s_[
                crop_i : crop_i + self.crop_size[0],
                crop_j : crop_j + self.crop_size[1],
            ]

            intensity_ll = intensity_ll[crop_slice]

        return video_name, intensity_ll
        
# Added class to accept bin files directly according to SPAD-MNIST folder structure
class IntensityCubeSimulated(Dataset):
    def __init__(
        self,
        photon_cube_location: str | list[str, ...],
        intensity_location: str | list[str, ...],
        reshape_size: Tuple[int, int] = None,
        crop_size: Tuple[int, int] = None,
        intensity_fps: int = 100,
        photon_cube_fps: int = 400,
        max_time_step: int = 1,
        oversampling: int = 1,
    ):
        self.oversampling = oversampling
        self.intensity_location = Path(intensity_location)
    
        if isinstance(photon_cube_location, str):
            photon_cube_location = [photon_cube_location]
            
        path_ll = []
        for data_dir in photon_cube_location:
            path_ll += list(Path(data_dir).rglob("*.bin"))
        
        if len(path_ll) == 0:
            raise ValueError(f"No .bin files found in {photon_cube_location}")
        
        self.path_ll = natsorted(path_ll)
    
    def __len__(self):
        return len(self.path_ll)
    
    def __getitem__(self, index):
        # load simulated frames
        path = self.path_ll[index]
        video_name = path.name

        raw_bytes = np.fromfile(path, dtype=np.uint8)

        unpacked_bits = np.unpackbits(raw_bytes)

        #TODO add pixels_per_frame as hydra config
        pixels_per_frame = 28 * 28
        num_frames = len(unpacked_bits) // pixels_per_frame

        cube_t_h_w = unpacked_bits.reshape(num_frames, 28, 28)

        photon_cube_np = np.transpose(cube_t_h_w, (1, 2, 0))

        photon_cube = torch.from_numpy(photon_cube_np).float()

        if self.oversampling > 1:
            photon_cube = repeat(
                photon_cube,
                "h w t -> h w (t num_repeat)",
                num_repeat = self.oversampling,
            )

        #load linear frames

        digit_folder = path.parent.name
        npy_filename = path.stem + ".npy"

        intensity_path = self.intensity_location / digit_folder / npy_filename

        if intensity_path.exists():
            intensity = np.load(intensity_path)
            intensity_ll = torch.from_numpy(intensity).float()
            
            # Stretch the 2D image to match the exact 3D length of the photon cube!
            final_time_steps = photon_cube.shape[2]
            intensity_ll = intensity_ll.unsqueeze(-1).expand(-1, -1, final_time_steps)
        else:
            raise FileNotFoundError(f"Could not find matching target: {intensity_path}")

        return video_name, photon_cube, intensity_ll



class IntensityImage(Dataset):
    def __init__(
        self,
        intensity_location: str | list[str, ...],
        reshape_size: Tuple[int, int] = None,
        crop_size: Tuple[int, int] = None,
    ):
        """
        :param intensity_location: The top-level directory containing the dataset
        """

        if isinstance(intensity_location, str):
            intensity_location = [intensity_location]

        # Collect all video files in all directories listed
        path_ll = []
        for video_dir in intensity_location:
            path_ll += list(Path(video_dir).glob("*.mp4"))

        # Load information about each video in the dataset.
        self.path_ll = sorted(path_ll)

        if crop_size is not None:
            crop_size = as_tuple(crop_size, size=2)
        self.crop_size = crop_size
        self.reshape_size = reshape_size

    def __len__(self):
        return len(self.path_ll)

    def __getitem__(self, index):
        path = self.path_ll[index]
        video_name = path.name
        intensity = get_intensity_cube(
            path, max_frames=1, reshape_size=self.reshape_size
        ).squeeze(-1)

        if self.crop_size:
            h, w = intensity.shape
            crop_i = randint(0, h - self.crop_size[0])
            crop_j = randint(0, w - self.crop_size[1])
            crop_slice = np.s_[
                crop_i : crop_i + self.crop_size[0],
                crop_j : crop_j + self.crop_size[1],
            ]

            intensity = intensity[crop_slice]

        return video_name, intensity


if __name__ == "__main__":
    data_folder = Path("/nobackup/vsundar4/xvfi/")
    assert data_folder.exists()

    data_kwargs = dict(
        intensity_location=data_folder / "train_16x_256_256",
        intensity_fps=16_000,
        photon_cube_fps=64_000,
        max_time_step=2048,
    )

    dataset = IntensityCube(**data_kwargs)
    dataloader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=1,
        num_workers=8,
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    for batch in tqdm(dataloader):
        video_name, intensity_ll = batch
        intensity_ll = intensity_ll.squeeze(0).to(device)
