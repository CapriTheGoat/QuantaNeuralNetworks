import json
import random
from math import ceil, floor
from pathlib import Path
from random import randint, shuffle

import OpenEXR
import imageio.v3 as iio
import numpy as np
import torch
import torchvision
from einops import rearrange, repeat
from jaxtyping import Float
from learned_projections.layers.counting import as_tuple

from quanta_neural_networks.reconstruction.dataloader import get_intensity_cube
from loguru import logger
from natsort import natsorted
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from tqdm import tqdm


# The following line only needs to run once for a user
# to download the necessary binaries to read HDR.
# imageio.plugins.freeimage.download()


def get_depth_cube(
    depth_path_ll: list[Path],
    max_frames: int,
    stride: int,
    reshape_size: int | tuple[int, int] = None,
    crop_size: int | tuple[int, int] = None,
    use_half: bool = False,
) -> Float[Tensor, "h w t"]:
    if reshape_size is not None:
        reshape_size = as_tuple(reshape_size, size=2)
    if crop_size is not None:
        crop_size = as_tuple(crop_size, size=2)

    depth_map_ll = []
    for e, depth_path in enumerate(depth_path_ll):
        if use_half and (e + 1) <= max_frames // 2:
            continue

        if (e + 1) % stride == 0:
            if depth_path.suffix == ".exr":
                with OpenEXR.File(str(depth_path)) as infile:
                    depth_map = infile.channels()["V"].pixels
            elif depth_path.suffix == ".hdr":
                depth_map = iio.imread(depth_path)[..., 0]
            else:
                raise NotImplementedError
            depth_map_ll.append(torch.from_numpy(depth_map))

        if e == max_frames - 1:
            break
    depth_map_ll: Float[Tensor, "h w t"] = torch.stack(depth_map_ll, dim=-1)

    if reshape_size:
        depth_map_ll = rearrange(depth_map_ll, "h w t -> t 1 h w")
        depth_map_ll = v2.functional.resize(
            depth_map_ll,
            size=reshape_size,
            interpolation=torchvision.transforms.InterpolationMode.NEAREST,
        )
        depth_map_ll = rearrange(depth_map_ll, "t 1 h w -> h w t")

    if crop_size:
        depth_map_ll = depth_map_ll[: crop_size[0], : crop_size[1]]

    return depth_map_ll


class BlenderDepth(Dataset):
    def __init__(
        self,
        dataset_location: str | Path,
        split: str,
        split_json: str = "split.json",
        refresh_split: bool = False,
        val_ratio: float = 1e-1,
        test_ratio: float = 1e-1,
        intensity_oversampling: int = 16,
        depth_extension: str = "exr",
        depth_subdir: str = "depths",
        max_intensity_frames: int = 2048,
        max_depth_frames: int = 64,
        depth_stride: int = 1,
        reshape_size: int | tuple[int, int] = None,
        crop_size: int | tuple[int, int] = None,
        use_half_gt: bool = True,
        save_depth_as_npy: bool = False,
        data_augment: bool = True,
    ):
        """
        :param dataset_location: The top-level directory containing the dataset
        """
        self.dataset_location = Path(dataset_location)
        self.max_intensity_frames = max_intensity_frames
        self.max_depth_frames = max_depth_frames
        self.reshape_size = reshape_size
        self.use_half_gt = use_half_gt
        self.depth_stride = depth_stride
        self.save_depth_as_npy = save_depth_as_npy
        self.depth_extension = depth_extension
        self.data_augment = data_augment

        if crop_size is not None:
            crop_size = as_tuple(crop_size, size=2)
        self.crop_size = crop_size

        if intensity_oversampling == 1:
            intensity_cube_name = "frames.mp4"
        elif intensity_oversampling == 4:
            intensity_cube_name = "frames_16kHz.mp4"
        elif intensity_oversampling == 16:
            intensity_cube_name = "frames_4kHz.mp4"
        else:
            raise NotImplementedError(
                f"To support {intensity_oversampling} subsampling, you may have to filter first with ffmpeg"
            )
        self.intensity_oversampling = intensity_oversampling

        split_json = self.dataset_location / split_json
        if not split_json.exists() or refresh_split:
            sequence_ll = [
                seq_dir
                for scene_dir in self.dataset_location.iterdir()
                if scene_dir.is_dir()
                for seq_dir in scene_dir.iterdir()
            ]

            # Ensure we have sufficient depth frames and an intensity cube
            sequence_ll = [
                sequence_dir
                for sequence_dir in sequence_ll
                if len(list((sequence_dir / depth_subdir).glob(f"*.{depth_extension}")))
                > max_depth_frames
                and (sequence_dir / intensity_cube_name).exists()
            ]

            assert 0 < val_ratio < 1

            shuffle(sequence_ll)

            train_ratio = 1 - val_ratio - test_ratio
            train_length = floor(train_ratio * len(sequence_ll))
            train_sequence_ll = sequence_ll[:train_length]

            val_length = floor(val_ratio * len(sequence_ll))
            val_sequence_ll = sequence_ll[train_length : val_length + train_length]

            test_sequence_ll = sequence_ll[train_length + val_length :]

            # Store relative paths
            split_json_dict = {
                "train": train_sequence_ll,
                "val": val_sequence_ll,
                "test": test_sequence_ll,
            }
            for k, path_ll in split_json_dict.items():
                split_json_dict[k] = [
                    str(path.relative_to(self.dataset_location)) for path in path_ll
                ]

            logger.info(
                f"Saving split with {len(train_sequence_ll)} train {len(val_sequence_ll)} val {len(test_sequence_ll)} test."
            )
            with open(split_json, "w") as f:
                json.dump(split_json_dict, f, sort_keys=True, indent=4)

        else:
            with open(split_json, "r") as f:
                split_json_dict = json.load(f)

        self.sequence_ll = natsorted(split_json_dict[split])
        self.sequence_ll = [self.dataset_location / path for path in self.sequence_ll]

        # Ensure we have sufficient depth
        self.sequence_ll = [
            sequence_dir
            for sequence_dir in self.sequence_ll
            if len(list((sequence_dir / depth_subdir).glob(f"*.{depth_extension}")))
            > self.max_depth_frames
            and (sequence_dir / intensity_cube_name).exists()
        ]
        self.intensity_cube_ll = [
            sequence_dir / intensity_cube_name for sequence_dir in self.sequence_ll
        ]
        self.depth_dir_ll = [
            sequence_dir / depth_subdir for sequence_dir in self.sequence_ll
        ]

    def __len__(self):
        return len(self.intensity_cube_ll)

    def __getitem__(self, index):
        sequence_dir = self.sequence_ll[index]
        sequence_name = f"{sequence_dir.parent.name}_{sequence_dir.name}"

        intensity_ll = get_intensity_cube(
            self.intensity_cube_ll[index],
            ceil(self.max_intensity_frames / self.intensity_oversampling),
            self.reshape_size,
        )

        intensity_ll = repeat(
            intensity_ll,
            "h w t -> h w (t num_repeat)",
            num_repeat=self.intensity_oversampling,
        )

        depth_dir = self.depth_dir_ll[index]
        depth_path_ll = sorted(list(depth_dir.glob(f"*.{self.depth_extension}")))

        h, w, _ = intensity_ll.shape
        npy_depth_file = (
            depth_dir
            / f"depth_[{h},{w},:{self.max_depth_frames}:{self.depth_stride}].npy"
        )

        if npy_depth_file.exists():
            depth_map_ll = torch.from_numpy(np.load(npy_depth_file))
        else:
            depth_map_ll = get_depth_cube(
                depth_path_ll,
                self.max_depth_frames,
                self.depth_stride,
                self.reshape_size,
                use_half=self.use_half_gt,
            )

        if self.save_depth_as_npy and not npy_depth_file.exists():
            np.save(str(depth_dir / npy_depth_file), depth_map_ll.numpy())

        # Scale from feet to meters
        if sequence_dir.parent.name == "bathroom2":
            depth_map_ll /= 3.28084

        # Crop depth and intensity
        if self.crop_size:
            h, w, _ = intensity_ll.shape
            crop_i = randint(0, h - self.crop_size[0])
            crop_j = randint(0, w - self.crop_size[1])
            crop_slice = np.s_[
                crop_i : crop_i + self.crop_size[0],
                crop_j : crop_j + self.crop_size[1],
            ]

            intensity_ll = intensity_ll[crop_slice]
            depth_map_ll = depth_map_ll[crop_slice]

        # Horz flipping
        if self.data_augment:
            if random.random() < 0.5:
                intensity_ll = torch.flip(intensity_ll, dims=[1])
                depth_map_ll = torch.flip(depth_map_ll, dims=[1])

        return sequence_name, intensity_ll, depth_map_ll


class XVFIDepth(Dataset):
    def __init__(
        self,
        intensity_location: str | Path,
        depth_location: str | Path,
        intensity_fps: int = 16_000,
        photon_cube_fps: int = 64_000,
        depth_subsampling: int = 64,
        max_intensity_frames: int = 2048,
        reshape_size: int | tuple[int, int] = None,
        crop_size: int | tuple[int, int] = None,
        use_half_gt: bool = True,
    ):
        """
        :param intensity_location: The top-level directory containing the dataset
        """
        self.intensity_location = Path(intensity_location)
        self.depth_location = Path(depth_location)

        self.intensity_cube_ll = sorted(list(self.intensity_location.glob("*.mp4")))

        self.depth_cube_ll = [
            self.depth_location / file.with_suffix(".npy").name
            for file in self.intensity_cube_ll
        ]

        self.photon_cube_oversampling = ceil(photon_cube_fps / intensity_fps)
        self.depth_subsampling = depth_subsampling
        self.max_time_step = max_intensity_frames
        self._max_intensity_frames = ceil(
            max_intensity_frames / self.photon_cube_oversampling
        )

        if crop_size:
            crop_size = as_tuple(crop_size, size=2)
        self.crop_size = crop_size

        if reshape_size:
            reshape_size = as_tuple(reshape_size, size=2)
        self.reshape_size = reshape_size
        self.use_half_gt = use_half_gt

    def __len__(self):
        return len(self.intensity_cube_ll)

    def __getitem__(self, index):
        sequence_name = self.intensity_cube_ll[index].stem

        # T, H, W
        depth_map_ll = torch.from_numpy(np.load(str(self.depth_cube_ll[index])))
        depth_map_ll = rearrange(depth_map_ll, "t h w -> h w t")

        stride = depth_map_ll.shape[-1] * self.depth_subsampling / self.max_time_step
        assert (
            stride >= 1
        ), "Stride < 1, which indicates that there aren't enough depth frames in the npy file."
        stride = int(stride)
        if stride > 0:
            depth_map_ll = depth_map_ll[..., stride - 1 :: stride]

        # Reshape depth maps
        if self.reshape_size:
            depth_map_ll = rearrange(depth_map_ll, "h w t -> t 1 h w")
            depth_map_ll = v2.functional.resize(
                depth_map_ll,
                size=list(self.reshape_size),
                interpolation=torchvision.transforms.InterpolationMode.NEAREST,
            )
            depth_map_ll = rearrange(depth_map_ll, "t 1 h w -> h w t")

        # Reshape to depth dims
        intensity_ll = get_intensity_cube(
            self.intensity_cube_ll[index],
            self._max_intensity_frames,
            tuple(depth_map_ll.shape[:2]),
        )

        intensity_ll = repeat(
            intensity_ll,
            "h w t -> h w (t num_repeat)",
            num_repeat=self.photon_cube_oversampling,
        )

        if self.use_half_gt:
            depth_map_ll = depth_map_ll[..., depth_map_ll.shape[-1] // 2 :]

        # Crop depth and intensity
        if self.crop_size:
            h, w, _ = intensity_ll.shape
            crop_i = randint(0, h - self.crop_size[0])
            crop_j = randint(0, w - self.crop_size[1])
            crop_slice = np.s_[
                crop_i : crop_i + self.crop_size[0],
                crop_j : crop_j + self.crop_size[1],
            ]

            intensity_ll = intensity_ll[crop_slice]
            depth_map_ll = depth_map_ll[crop_slice]

        return sequence_name, intensity_ll, depth_map_ll


if __name__ == "__main__":
    # On oldfashioned
    for dataset in [
        # XVFIDepth(
        #     intensity_location=Path("/nobackup2/shared/xvfi_16x/train_16x"),
        #     depth_location=Path(
        #         "/nobackup2/vsundar4/xvfi_depth_pro_1000_fps/train_512_512"
        #     ),
        #     depth_subsampling=128,
        # ),
        BlenderDepth(
            dataset_location=Path("/nobackup/vsundar4/blender_depth_faster/"),
            split="test",
            test_ratio=0.15,
            val_ratio=0.15,
            reshape_size=(224, 224),
            depth_stride=8,
            refresh_split=False,
            max_depth_frames=64,
            max_intensity_frames=4096,
            depth_extension="exr",
            intensity_oversampling=4,
        ),
    ]:
        dataloader = DataLoader(
            dataset,
            shuffle=True,
            batch_size=4,
            num_workers=8,
        )
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        min_depth = None
        max_depth = None
        pbar = tqdm(dataset)
        for e, batch in enumerate(pbar):
            sequence_name, intensity_ll, depth_map_ll = batch

            min_depth = (
                min(depth_map_ll.min(), min_depth)
                if min_depth is not None
                else depth_map_ll.min()
            )
            max_depth = (
                max(depth_map_ll.max(), max_depth)
                if max_depth is not None
                else depth_map_ll.max()
            )

            pbar.set_description(
                f"Min depth {min_depth:.3g} max depth {max_depth:.3g} sample median {torch.median(depth_map_ll).item()}"
            )
