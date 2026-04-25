"""
Dataset for point tracking training and evaluation.

This module provides a PyTorch dataset for loading point tracking data,
including intensity sequences, tracked point trajectories, and occlusion masks.
"""

import random
from math import ceil
from pathlib import Path

import numpy as np
import torch
from einops import repeat, rearrange
from jaxtyping import Float
from loguru import logger
from natsort import natsorted
from torch import Tensor
from torch.utils.data import Dataset

from quanta_neural_networks.reconstruction.dataloader import get_intensity_cube


class TrackingDataset(Dataset):
    """
    Dataset for point tracking with intensity sequences and trajectories.
    
    This dataset loads intensity sequences, tracked point trajectories, and
    occlusion masks for training point tracking models.
    
    Outputs:
    - binary sequence (from 16X interpolation): (T*16, 256, 256, 3)
    - tracked points: (N, T, 2) 
    - occluded: occlusion mask of shape (N,T)
    
    Here we use T = 128
    
    Example: video, target_points, occluded
    torch.Size([1, 128, 3, 256, 256]) torch.Size([1, 3138, 128, 2]) torch.Size([1, 3138, 128])
    """
    def __init__(
        self,
        dataset_location: str | Path,
        split: str = "train",
        use_half_gt: bool = True,
        intensity_fps: int = 16_000,
        photon_cube_fps: int = 64_000,
        trajectory_fps: int = 1_000,
        max_time_step: int = 2048,
        max_points: int | None = None,
        trajectory_subsampling: int = 64,
        mix_random_points: bool = True,
        data_augment: bool = True,
    ) -> None:
        """
        Initialize the tracking dataset.
        
        :param dataset_location: Path to the dataset directory
        :param split: Dataset split ('train' or 'val')
        :param use_half_gt: Whether to use only the second half of trajectories
        :param intensity_fps: Frames per second for intensity data
        :param photon_cube_fps: Frames per second for photon cube data
        :param trajectory_fps: Frames per second for trajectory data
        :param max_time_step: Maximum number of time steps
        :param max_points: Maximum number of points to track (None for all)
        :param trajectory_subsampling: Subsampling factor for trajectories
        :param mix_random_points: If True, mix randomly chosen points in addition to
            points with highest movement. Else: choose only points with highest
            movement (top-k)
        :param data_augment: Whether to apply data augmentation
        """
        self.dataset_location = Path(dataset_location)
        self.use_half_gt = use_half_gt

        self.photon_cube_oversampling = ceil(photon_cube_fps / intensity_fps)
        self.max_time_step = max_time_step
        self._max_intensity_frames = ceil(max_time_step / self.photon_cube_oversampling)
        self.max_points = max_points

        self._trajectory_subsampling = int(
            trajectory_subsampling / (photon_cube_fps / trajectory_fps)
        )
        self._max_trajectory_step = ceil(
            max_time_step / (photon_cube_fps / trajectory_fps)
        )

        assert split in ["train", "val"]

        self.intensity_cube_ll = natsorted(
            list((Path(dataset_location) / split).rglob("sequence_16kHz.mp4"))
        )
        self.npz_path_ll = [
            path.parent / "points.npz" for path in self.intensity_cube_ll
        ]

        # Augmentation
        self.horz_flip_prob = 1 / 8
        self.vertical_flip_prob = 1 / 8
        self.transpose_prob = 1 / 2
        self.reverse_prob = 1 / 8

        self.mix_random_points = mix_random_points
        self.data_augment = data_augment

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        
        :return: Number of samples
        """
        return len(self.intensity_cube_ll)

    def __getitem__(self, idx: int) -> tuple[str, Float[Tensor, "height width time"], Float[Tensor, "num_frame num_points coords"], Float[Tensor, "num_frame num_points"]]:
        """
        Get a sample from the dataset.
        
        :param idx: Sample index
        :return: Tuple of (sequence_name, intensity_sequence, trajectory, unoccluded_mask)
        """
        sequence_name = self.intensity_cube_ll[idx].parent.name

        intensity_ll = get_intensity_cube(
            self.intensity_cube_ll[idx],
            self._max_intensity_frames,
        )

        intensity_ll = repeat(
            intensity_ll,
            "h w t -> h w (t num_repeat)",
            num_repeat=self.photon_cube_oversampling,
        )

        # Load tracked points and metadata
        point_data = np.load(str(self.npz_path_ll[idx]))

        trajectory: Float[Tensor, "num_frame num_points coords"] = torch.from_numpy(
            rearrange(
                point_data["target_points"],
                "num_points num_frame coords -> num_frame num_points coords",
            )
        )

        unoccluded = torch.logical_not(
            torch.from_numpy(
                rearrange(
                    point_data["occluded"],
                    "num_points num_frame -> num_frame num_points",
                )
            )
        )

        trajectory_frame_slice = np.s_[
            : self._max_trajectory_step : self._trajectory_subsampling
        ]
        trajectory = trajectory[trajectory_frame_slice]
        unoccluded = unoccluded[trajectory_frame_slice]

        if self.use_half_gt:
            half_slice = np.s_[len(trajectory) // 2 :]
            trajectory = trajectory[half_slice]
            unoccluded = unoccluded[half_slice]

        # Point should be unoccluded at start
        valid_points = unoccluded[0] > 0
        trajectory = trajectory[:, valid_points]
        unoccluded = unoccluded[:, valid_points]

        if unoccluded.shape[1] == 0:
            logger.error(f"No valid points in file {self.npz_path_ll[idx]}")
            return self.__getitem__(random.randint(0, len(self) - 1))

        if self.max_points:
            trajectory_deviation = (trajectory[-1] - trajectory[0]).abs().sum(dim=-1)

            # Sort and choose
            trajectory_deviation_argsort = torch.argsort(
                trajectory_deviation, descending=True
            )

            if self.mix_random_points:
                trajectory_sort_slice = trajectory_deviation_argsort[
                    : self.max_points // 2
                ]
                trajectory_left_out = trajectory_deviation_argsort[
                    self.max_points // 2 :
                ]

                # Choose randomly from remaining points
                random_slice = torch.randperm(len(trajectory_left_out))[
                    : self.max_points // 2
                ]
                trajectory_points_slice = torch.cat(
                    (
                        trajectory_sort_slice,
                        trajectory_left_out[random_slice],
                    )
                )

            else:
                trajectory_points_slice = trajectory_deviation_argsort[
                    : self.max_points
                ]
            trajectory = trajectory[:, trajectory_points_slice]
            unoccluded = unoccluded[:, trajectory_points_slice]

        if self.data_augment:
            # Randomly perm points
            perm = torch.randperm(trajectory.shape[1])
            trajectory = trajectory[:, perm]
            unoccluded = unoccluded[:, perm]

            # Aug
            if random.random() < self.reverse_prob:
                intensity_ll = torch.flip(intensity_ll, dims=(-1,))
                trajectory = torch.flip(trajectory, dims=(0,))
                unoccluded = torch.flip(unoccluded, dims=(0,))

            if random.random() < self.transpose_prob:
                intensity_ll = rearrange(intensity_ll, "h w t -> w h t")

                # Swap x and y
                trajectory = torch.flip(trajectory, dims=(2,))

            if random.random() < self.horz_flip_prob:
                intensity_ll = torch.flip(intensity_ll, dims=(1,))
                trajectory[:, :, 0] = intensity_ll.shape[1] - trajectory[:, :, 0]

            if random.random() < self.vertical_flip_prob:
                intensity_ll = torch.flip(intensity_ll, dims=(0,))
                trajectory[:, :, 1] = intensity_ll.shape[0] - trajectory[:, :, 1]

            # Shuffle points order. Else they appear in order of motion
            points_slice = torch.randperm(trajectory.shape[1]).long()
            trajectory = trajectory[:, points_slice]
            unoccluded = unoccluded[:, points_slice]

        return sequence_name, intensity_ll, trajectory, unoccluded
