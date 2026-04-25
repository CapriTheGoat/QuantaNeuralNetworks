"""
Dense network with space-time factorization. Implementation motivated by:
https://arxiv.org/pdf/2305.10006 (EfficientSSD).
"""

from typing import Callable

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Bool, Float
from torch import nn, Tensor
from torch.nn import functional as F
from loguru import logger
from quanta_neural_networks.integrator import PerPixelBayesian
from quanta_neural_networks.ssd import SSD


def get_norm_fn(norm_fn: str) -> Callable:
    if norm_fn == "batch_norm":
        return nn.BatchNorm2d
    elif norm_fn == "instance_norm":
        return nn.InstanceNorm2d
    else:
        return nn.Identity


class DenseLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        growth_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        norm_fn: str = None,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels, growth_channels, kernel_size=kernel_size, padding=padding
        )
        self.norm_fn = get_norm_fn(norm_fn)(growth_channels)

    def forward(self, x):
        squeeze_output = False
        if x.ndim == 3:
            x = x.unsqueeze(0)
            squeeze_output = True

        out = F.relu(self.norm_fn(self.conv(x)))
        out = torch.cat([x, out], dim=1)
        if squeeze_output:
            out = out.squeeze(0)

        return out


class DenseBlock(nn.Module):
    def __init__(self, dim: int, state_dim: int):
        super().__init__()
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
        )

        self.ssd = SSD(
            in_dim=dim,
            head_dim=dim // 4,
            state_dim=state_dim,
            identity_init=False,
        )

        self.feedforward = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1),
        )

    def clear_hidden_state(self):
        self.ssd.clear_hidden_state()

    def forward(
        self, x: Float[Tensor, "seq_len channels height width"], t_index_ll: list[int]
    ) -> tuple[Float[Tensor, "seq_len channels height width"], list[int]]:
        spatial_out = self.spatial_conv(x)
        temporal_out, t_index_ll = self.ssd(x, t_index_ll)

        feedforward_in = spatial_out + temporal_out + x
        feedforward_out = self.feedforward(feedforward_in) + feedforward_in

        return feedforward_out, t_index_ll

    def forward_online(
        self, x: Float[Tensor, "channels height width"], time_instant: int
    ) -> Float[Tensor, "channels height width"]:
        spatial_out = self.spatial_conv(x)
        temporal_out = self.ssd.forward_online(x, time_instant=time_instant)

        feedforward_in = spatial_out + temporal_out + x
        feedforward_out = self.feedforward(feedforward_in) + feedforward_in

        return feedforward_out


class ResidualDenseBlock(nn.Module):
    def __init__(self, dim: int, group_num: int, state_dim: int):
        super().__init__()
        self.dense_block_ll = nn.ModuleList()
        self.group_num = group_num

        group_dim = dim // group_num
        self.dense_conv_ll = nn.ModuleList()

        for i in range(group_num):
            self.dense_block_ll.append(DenseBlock(group_dim, state_dim))
            if i > 0:
                self.dense_conv_ll.append(
                    nn.Sequential(
                        nn.Conv2d(group_dim * (i + 1), group_dim, kernel_size=1),
                        nn.LeakyReLU(inplace=True),
                    )
                )
        self.last_conv = nn.Conv2d(dim, dim, kernel_size=1)

    def clear_hidden_state(self):
        for dense_block in self.dense_block_ll:
            dense_block.clear_hidden_state()

    def forward(
        self, x: Float[Tensor, "seq_len channels height width"], t_index_ll: list[int]
    ) -> tuple[Float[Tensor, "seq_len channels height width"], list[int]]:
        input_ll = torch.chunk(x, chunks=self.group_num, dim=1)

        dense_in = input_ll[0]
        out_list = []
        dense_out, t_index_ll = self.dense_block_ll[0](dense_in, t_index_ll)
        out_list.append(dense_out)

        for i in range(1, self.group_num):
            in_list = out_list.copy()
            in_list.append(input_ll[i])
            dense_in = torch.cat(in_list, dim=1)
            dense_in = self.dense_conv_ll[i - 1](dense_in)
            dense_out, t_index_ll = self.dense_block_ll[i](dense_in, t_index_ll)
            out_list.append(dense_out)

        out = torch.cat(out_list, dim=1)
        out = self.last_conv(out)
        out = x + out
        return out, t_index_ll

    def forward_online(
        self, x: Float[Tensor, "channels height width"], time_instant: int
    ) -> Float[Tensor, "channels height width"]:
        input_ll = torch.chunk(x, chunks=self.group_num, dim=0)

        dense_in = input_ll[0]
        out_list = []
        dense_out = self.dense_block_ll[0].forward_online(
            dense_in, time_instant=time_instant
        )
        out_list.append(dense_out)

        for i in range(1, self.group_num):
            in_list = out_list.copy()
            in_list.append(input_ll[i])
            dense_in = torch.cat(in_list, dim=0)
            dense_in = self.dense_conv_ll[i - 1](dense_in)
            dense_out = self.dense_block_ll[i].forward_online(
                dense_in, time_instant=time_instant
            )
            out_list.append(dense_out)

        out = torch.cat(out_list, dim=0)
        out = self.last_conv(out)
        out = x + out
        return out


class EfficientSSD(nn.Module):
    """
    EfficientSSD model, as in the original paper https://arxiv.org/pdf/2305.10006

    :param channels: Channel dimension for inner layers
    :param state_dim: State dim for SSD
    :param subsampling: Subsampling interval
    :param group_num: Number of dense groups
    :param units: Number of residual dense units
    :param integrator_kwargs: Args for the integrator

    .. note::
        Inspired by EfficientSSD (arxiv:2305.10006).
    """
    def __init__(
        self,
        channels: int = 64,
        state_dim: int = 12,
        subsampling: int | tuple[int, int] = 64,
        group_num: int = 8,
        units: int = 6,
        **integrator_kwargs,
    ):
        """

        :param channels:
        :param state_dim:
        :param subsampling: fixed number or an interval (for irregular inference)
        :param group_num:
        :param units:
        :param integrator:
        :param integrator_kwargs:
        """
        self.subsampling = subsampling
        self.units = units
        super().__init__()
        self.integrator = PerPixelBayesian(**integrator_kwargs)
        logger.info(f"Using integrator {self.integrator} with EfficentSSD")

        self.stem = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=7, stride=1, padding=3),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels, channels * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels * 2, channels * 4, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
        )

        self.ssd_stem = SSD(
            in_dim=channels * 4,
            state_dim=state_dim,
            head_dim=channels,
            identity_init=False,
        )

        self.upscale = nn.Sequential(
            nn.Conv2d(channels * 4, channels * 8, kernel_size=1),
            nn.PixelShuffle(2),
            nn.Conv2d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels * 2, channels, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1),
        )

        self.residual_dense_ll = nn.ModuleList()
        for i in range(self.units):
            self.residual_dense_ll.append(
                ResidualDenseBlock(
                    channels * 4,
                    group_num=group_num,
                    state_dim=state_dim,
                )
            )

        self.t_prev = 0

    def reset(self, recurse=True):
        super().reset(recurse)
        self.clear_hidden_state()
        self.t_prev = 0

    def clear_hidden_state(self):
        self.t_prev = 0
        self.ssd_stem.clear_hidden_state()
        for residual_dense in self.residual_dense_ll:
            residual_dense.clear_hidden_state()

    def forward(
        self,
        photon_cube: Float[Tensor, "h w t"],
        bocpd_gamma: float = 1e-4,
        hot_pixel_mask: Bool[np.ndarray, "h w"] = None,
    ) -> tuple[Float[Tensor, "h w seq_len"], list[int]]:
        """
        Forward pass to reconstruct with EfficientSSD.

        :param photon_cube: Input photon event tensor, Float[Tensor, "height width time"]
        :param bocpd_gamma: Integrator argument
        :param hot_pixel_mask: Mask for hot/dead pixels
        :return: Tuple (reconstruction, time_indices)
        """
        height, width, t = photon_cube.shape
        t_index_ll = np.arange(1, t + 1)

        x_ll = self.integrator.process_photon_cube(
            photon_cube,
            bocpd_gamma=bocpd_gamma,
            hot_pixel_mask=hot_pixel_mask,
            subsampling=self.subsampling,
        )
        x_ll = rearrange(x_ll, "h w t -> t 1 h w")
        t_index_ll = t_index_ll[self.subsampling - 1 :: self.subsampling]

        out_ll = self.stem(x_ll)
        out_ll, t_index_ll = self.ssd_stem(out_ll, t_index_ll)

        for e, resdnet in enumerate(self.residual_dense_ll):
            out_ll, t_index_ll = resdnet(out_ll, t_index_ll)

        out_ll = self.upscale(out_ll)
        out_ll = rearrange(out_ll, "t 1 h w -> h w t")

        return out_ll, t_index_ll

    def forward_online(
        self,
        photon_cube: Float[Tensor, "h w t"],
        bocpd_gamma: float = 1e-4,
        hot_pixel_mask: Bool[np.ndarray, "h w"] = None,
        clear_states: bool = True
    ) -> tuple[Float[Tensor, "h w seq_len"], list[int]]:
        height, width, t = photon_cube.shape

        # Clear states
        if clear_states:
            self.clear_hidden_state()
        t_index_ll = np.arange(1 + self.t_prev, t + 1 + self.t_prev)

        x_ll = self.integrator.process_photon_cube(
            photon_cube,
            bocpd_gamma=bocpd_gamma,
            hot_pixel_mask=hot_pixel_mask,
            subsampling=self.subsampling,
            clear_states=clear_states,
        )
        x_ll = rearrange(x_ll, "h w t -> t 1 h w")
        t_index_ll = t_index_ll[
            self.subsampling - 1 :: self.subsampling
        ]

        out_t_index_ll = []
        out_ll = []

        for recons_idx, t_index in enumerate(t_index_ll):
            x = x_ll[recons_idx]

            if not t_index % self.subsampling == 0:
                continue

            out = self.stem(x)
            out: Float[Tensor, "channels h w"] = self.ssd_stem.forward_online(
                out, time_instant=t_index
            )

            for e, resdnet in enumerate(self.residual_dense_ll):
                out = resdnet.forward_online(out, time_instant=t_index)

            out_t_index_ll.append(t_index)
            out = self.upscale(out)

            out_ll.append(out.squeeze(0))

        self.t_prev += t
        out_ll = torch.stack(out_ll, dim=-1)

        return out_ll, out_t_index_ll


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    h, w, t = 96, 96, 2048
    bocpd_gamma = 3e-4
    photon_cube = torch.rand(h, w, t, device=device)
    model = EfficientSSD().to(device)
    model.eval()

    out_online, out_online_t_index_ll = model.forward_online(
        photon_cube, bocpd_gamma=bocpd_gamma
    )
    out_online.sum().backward()
    logger.info(f"Output online shape {out_online.shape}")
