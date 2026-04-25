"""
Implements PIPS++ point tracking core. Source: https://arxiv.org/abs/2204.00262
"""
from typing import Tuple

import numpy as np
import torch
from einops import rearrange, repeat
from jaxtyping import Bool, Float, Int
from loguru import logger
from torch import nn, Tensor
from torch.nn import functional as F

from quanta_neural_networks.integrator import PerPixelBayesian
from quanta_neural_networks.point_tracking.correlation import CorrelationBlock
from quanta_neural_networks.point_tracking.utils import (
    bilinear_sample2d,
    fourier_position_embed_xy,
)
from quanta_neural_networks.ssd import SSD


class Conv1dPad(nn.Module):
    """
    Conv1d with auto-computed padding ("same" padding)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
        )

    def forward(self, x):
        p = max(0, self.kernel_size - 1)
        pad_left = p // 2
        pad_right = p - pad_left
        out = F.pad(x, (pad_left, pad_right), "constant", 0)
        out = self.conv(out)
        return out


class ResidualBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        is_first_block=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.is_first_block = is_first_block

        self.norm1 = nn.InstanceNorm1d(in_channels)
        self.conv1 = Conv1dPad(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size
        )

        self.norm2 = nn.InstanceNorm1d(out_channels)
        self.conv2 = Conv1dPad(
            in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size
        )

    def forward(self, x):
        identity = x

        out = x
        if not self.is_first_block:
            out = F.relu(self.norm1(out))

        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)

        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1, -2)
            ch1 = (self.out_channels - self.in_channels) // 2
            ch2 = self.out_channels - self.in_channels - ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1, -2)

        out += identity
        return out


class DeltaBlock(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        corr_levels: int = 4,
        corr_radius: int = 3,
        num_block: int = 8,
        kernel_size: int = 3,
        increase_filter_gap: int = 2,
    ):
        super(DeltaBlock, self).__init__()

        in_channels = (corr_levels * (2 * corr_radius + 1) ** 2) * 3 + latent_dim + 2

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.first_block_conv = Conv1dPad(
            in_channels=in_channels,
            out_channels=latent_dim,
            kernel_size=kernel_size,
        )
        self.first_block_norm = nn.InstanceNorm1d(latent_dim)
        out_channels = latent_dim

        self.basicblock_list = nn.ModuleList()
        for i_block in range(num_block):
            if i_block == 0:
                in_channels = latent_dim
                out_channels = in_channels
            else:
                in_channels = int(
                    latent_dim * 2 ** ((i_block - 1) // increase_filter_gap)
                )
                if (i_block % increase_filter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels

            self.basicblock_list.append(
                ResidualBlock1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    is_first_block=i_block == 0,
                )
            )

        self.final_norm = nn.InstanceNorm1d(out_channels)
        self.dense = nn.Linear(out_channels, 2)

    def forward(self, fcorr, flow):
        flow_sincos = fourier_position_embed_xy(flow, self.latent_dim)

        x: Float[Tensor, "num_frame batch num_coords"] = torch.cat(
            [fcorr, flow_sincos], dim=2
        )

        x = rearrange(x, "num_frame batch num_coords -> batch num_coords num_frame")

        # conv1d wants channels in the middle
        x = F.relu(self.first_block_conv(x))
        for block in self.basicblock_list:
            x = block(x)
        x = rearrange(F.relu(x), "batch channels num_frame -> num_frame batch channels")

        delta_coords = self.dense(x)
        return delta_coords


class ResidualBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ):
        super(ResidualBlock2d, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, stride=stride
        )
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.norm2 = nn.InstanceNorm2d(out_channels)

        if stride == 1:
            self.downsample = None
        else:
            self.norm3 = nn.InstanceNorm2d(out_channels)
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                self.norm3,
            )

    def forward(self, x):
        y = x
        y = F.relu(self.norm1(self.conv1(y)))
        y = F.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return F.relu(x + y)


class FeatureEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        output_dim=128,
        layer_channels: tuple[int, ...] = (64, 96, 128, 128),
        spatial_stride: int = 4,
    ):
        super().__init__()

        self.input_dim = input_dim

        self.conv1 = nn.Conv2d(
            input_dim,
            layer_channels[0],
            kernel_size=7,
            stride=2,
            padding=3,
        )
        self.conv2 = nn.Conv2d(
            sum(layer_channels), output_dim * 2, kernel_size=3, padding=1
        )

        self.norm1 = nn.InstanceNorm2d(layer_channels[0])
        self.norm2 = nn.InstanceNorm2d(output_dim * 2)
        self.spatial_stride = spatial_stride
        self.layer_channels = layer_channels

        in_channels_ll = [layer_channels[0], *layer_channels[:3]]
        self.in_channels_ll = in_channels_ll
        out_channels_ll = layer_channels
        stride_ll = [1, 2, 2, 2]

        self.layer1 = self._make_layer(
            in_channels_ll[0], out_channels_ll[0], stride_ll[0]
        )
        self.layer2 = self._make_layer(
            in_channels_ll[1], out_channels_ll[1], stride_ll[1]
        )
        self.layer3 = self._make_layer(
            in_channels_ll[2], out_channels_ll[2], stride_ll[2]
        )
        self.layer4 = self._make_layer(
            in_channels_ll[3], out_channels_ll[3], stride_ll[3]
        )

        self.conv3 = nn.Conv2d(output_dim * 2, output_dim, kernel_size=1)

    def _make_layer(
        self, in_channels: int, out_channels: int, stride: int = 1
    ) -> nn.Sequential:
        layer_ll = [
            ResidualBlock2d(in_channels, out_channels, stride=stride),
            ResidualBlock2d(out_channels, out_channels, stride=1),
        ]

        layer_ll = nn.Sequential(*layer_ll)

        # Tack on some attributes
        layer_ll.in_channels = in_channels
        layer_ll.out_channels = out_channels

        return layer_ll

    def _interpolate(
        self,
        tensor: Float[Tensor, "batch channels height width"],
        size: tuple[int, int],
    ):
        squeeze_output = False
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
            squeeze_output = True
        output = F.interpolate(
            tensor,
            size=size,
            mode="bilinear",
            align_corners=True,
        )
        if squeeze_output:
            output = output.squeeze(0)
        return output

    def forward(
        self,
        video: Float[Tensor, "h w t"],
    ) -> Float[Tensor, "num_frame channels height width"]:
        height, width, _ = video.shape

        # Repeat along channel dim, normalize to -1...1
        x_ll = repeat(video, "h w t -> t c h w", c=self.input_dim)

        x_ll = 2 * x_ll - 1

        x_ll = F.relu(self.norm1(self.conv1(x_ll)))
        a_ll = self.layer1(x_ll)
        b_ll = self.layer2(a_ll)
        c_ll = self.layer3(b_ll)
        d_ll = self.layer4(c_ll)

        interpolate_size = (height // self.spatial_stride, width // self.spatial_stride)
        a_ll = self._interpolate(a_ll, interpolate_size)

        b_ll = self._interpolate(b_ll, interpolate_size)

        c_ll = self._interpolate(c_ll, interpolate_size)

        d_ll = self._interpolate(d_ll, interpolate_size)

        out_ll = torch.cat([a_ll, b_ll, c_ll, d_ll], dim=1)
        out_ll = F.relu(self.norm2(self.conv2(out_ll)))
        out_ll = self.conv3(out_ll)

        return out_ll


class PhotonEncoder(FeatureEncoder):
    def __init__(
        self,
        input_dim: int = 3,
        state_dim: int = 8,
        output_dim: int = 128,
        subsampling: int | tuple[int, int] = 64,
        layer_channels: tuple[int, ...] = (64, 96, 128, 128),
        spatial_stride: int = 4,
        **integrator_kwargs,
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            layer_channels=layer_channels,
            spatial_stride=spatial_stride,
        )

        self.integrator = PerPixelBayesian(
            **integrator_kwargs, normalize=True, subsampling=subsampling
        )
        logger.info(f"Using integrator {self.integrator} with ResNet (pips) encoder.")

        self.subsampling = subsampling

        self.ssd_ll = nn.ModuleList(
            SSD(
                state_dim=state_dim,
                in_dim=self.in_channels_ll[ssm_idx],
                head_dim=self.in_channels_ll[ssm_idx] // 4,
                identity_init=True,
            )
            for ssm_idx in range(len(self.in_channels_ll))
        )

    def forward(
        self,
        photon_cube: Float[Tensor, "h w t"],
        bocpd_gamma: float = 1e-4,
        hot_pixel_mask: Bool[np.ndarray, "h w"] = None,
        **integrator_kwargs,
    ) -> tuple[Float[Tensor, "num_frame channels height width"], list[int]]:
        height, width, t = photon_cube.shape
        t_index_ll = list(range(1, t + 1))

        x_ll = self.integrator.process_photon_cube(
            photon_cube,
            bocpd_gamma=bocpd_gamma,
            hot_pixel_mask=hot_pixel_mask,
            **integrator_kwargs,
        )
        x_ll = rearrange(x_ll, "h w t -> t 1 h w")
        t_index_ll = t_index_ll[self.subsampling - 1 :: self.subsampling]

        # Repeat along channel dim, normalize to -1...1
        x_ll = repeat(x_ll, "t 1 h w -> t c h w", c=self.input_dim)
        x_ll = 2 * x_ll - 1

        x_ll = F.relu(self.norm1(self.conv1(x_ll)))

        x_ll, t_index_ll = self.ssd_ll[0](x_ll, t_index_ll)
        a_ll = self.layer1(x_ll)

        b_ll, t_index_ll = self.ssd_ll[1](a_ll, t_index_ll)
        b_ll = self.layer2(b_ll)

        c_ll, t_index_ll = self.ssd_ll[2](b_ll, t_index_ll)
        c_ll = self.layer3(c_ll)

        d_ll, t_index_ll = self.ssd_ll[3](c_ll, t_index_ll)
        d_ll = self.layer4(d_ll)

        interpolate_size = (height // self.spatial_stride, width // self.spatial_stride)
        stride_a = a_ll.shape[0] // d_ll.shape[0]
        a_ll = self._interpolate(a_ll, interpolate_size)[stride_a - 1 :: stride_a]

        stride_b = b_ll.shape[0] // d_ll.shape[0]
        b_ll = self._interpolate(b_ll, interpolate_size)[stride_b - 1 :: stride_b]

        stride_c = c_ll.shape[0] // d_ll.shape[0]
        c_ll = self._interpolate(c_ll, interpolate_size)[stride_c - 1 :: stride_c]

        d_ll = self._interpolate(d_ll, interpolate_size)

        out_ll = torch.cat([a_ll, b_ll, c_ll, d_ll], dim=1)
        out_ll = F.relu(self.norm2(self.conv2(out_ll)))
        out_ll = self.conv3(out_ll)

        return out_ll, t_index_ll

    @property
    def first_layer_subsampling(self) -> int:
        return (
            self.subsampling
            if isinstance(self.subsampling, int)
            else min(self.subsampling)
        )

    def forward_online(
        self,
        photon_cube: Float[Tensor, "h w t"],
        bocpd_gamma: float = 1e-4,
        clear_states: bool = True,
        hot_pixel_mask: Bool[np.ndarray, "h w"] = None,
        **integrator_kwargs,
    ) -> tuple[Float[Tensor, "num_frame channels height width"], list[int]]:
        height, width, t = photon_cube.shape

        # Clear states
        # t_prev is useful when state is not cleared and we would like to resume inference
        if clear_states:
            for ssd in self.ssd_ll:
                ssd.clear_hidden_state()

        t_index_ll = list(range(1, t + 1))

        x_ll = self.integrator.process_photon_cube(
            photon_cube,
            bocpd_gamma=bocpd_gamma,
            hot_pixel_mask=hot_pixel_mask,
            subsampling=self.first_layer_subsampling,
            **integrator_kwargs,
        )
        t_index_ll = t_index_ll[
            self.first_layer_subsampling - 1 :: self.first_layer_subsampling
        ]

        out_t_index_ll = []
        out_ll = []

        interpolate_size = (height // self.spatial_stride, width // self.spatial_stride)

        # Useful for CPU runs

        for recons_idx, t_index in enumerate(t_index_ll):
            x = x_ll[..., recons_idx]

            if not t_index % self.subsampling == 0:
                continue

            # Repeat and normalize across channel dim
            x = repeat(x, "h w -> c h w", c=self.input_dim)
            x = 2 * x - 1

            x = F.relu(self.norm1(self.conv1(x)))

            x = self.ssd_ll[0].forward_online(x, time_instant=t_index)
            a = self.layer1(x)

            b = self.ssd_ll[1].forward_online(a, time_instant=t_index)
            b = self.layer2(b)

            c = self.ssd_ll[2].forward_online(b, time_instant=t_index)
            c = self.layer3(c)

            d = self.ssd_ll[3].forward_online(c, time_instant=t_index)
            d = self.layer4(d)

            a = self._interpolate(a, interpolate_size)
            b = self._interpolate(b, interpolate_size)
            c = self._interpolate(c, interpolate_size)
            d = self._interpolate(d, interpolate_size)

            out = torch.cat([a, b, c, d], dim=0)
            out = F.relu(self.norm2(self.conv2(out)))
            out = self.conv3(out)

            out_ll.append(out)
            out_t_index_ll.append(t_index)

        out_tensor: Float[Tensor, "num_frame channels height width"] = torch.stack(
            out_ll, dim=0
        )

        return out_tensor, out_t_index_ll


class PointTracker(nn.Module):
    """
    Point tracker module, inspired by/derived from PIPS++ methodology.

    :param state_dim: State dimension for sequential modeling
    :param subsampling: Frame subsampling factor
    :param corr_level: Correlation feature levels
    :param corr_radius: Window radius in correlation-block
    :param spatial_stride: Feature stride
    :param latent_dim: Latent dimension
    :param num_delta_block: Number of delta blocks
    :param delta_kernel_size: Delta block kernel size
    :param delta_hidden_dim: Hidden dim for delta blocks
    :param num_pips_iter: Number of PIPS iterations
    :param integrator_kwargs: Extra integrator args

    .. note::
        Some blocks inspired by the PIPS++ paper and project:
        https://arxiv.org/abs/2204.00262.
    """

    def __init__(
        self,
        state_dim: int = 8,
        subsampling: int = 64,
        corr_level: int = 4,
        corr_radius: int = 3,
        spatial_stride: int = 8,
        latent_dim: int = 128,
        num_delta_block: int = 3,
        delta_kernel_size: int = 3,
        delta_hidden_dim: int = 256,
        num_pips_iter: int = 3,
        **integrator_kwargs,
    ):
        super().__init__()

        self.corr_levels = corr_level
        self.corr_radius = corr_radius
        self.argmax_radius = 5  # pixels by spatial stride
        self.spatial_stride = spatial_stride
        self.num_pips_iter = num_pips_iter
        self.subsampling = subsampling

        self.fnet = PhotonEncoder(
            state_dim=state_dim,
            subsampling=subsampling,
            spatial_stride=spatial_stride,
            **integrator_kwargs,
        )

        self.delta_block = DeltaBlock(
            corr_levels=self.corr_levels,
            corr_radius=self.corr_radius,
            num_block=num_delta_block,
            hidden_dim=delta_hidden_dim,
            latent_dim=latent_dim,
            kernel_size=delta_kernel_size,
        )

    def pred_tracks(
        self,
        feature_map_ll: Float[Tensor, "num_frame channels height width"],
        coords_init: Int[Tensor, "num_points num_coords"],
        spatial_stride: int,
        beautify: bool = False,
    ) -> Float[Tensor, "num_preds num_frame num_points num_coords"]:
        feature_correlation_func = CorrelationBlock(
            feature_map_ll, num_levels=self.corr_levels, radius=self.corr_radius
        )
        feature_correlation_t_minus_2_func = CorrelationBlock(
            feature_map_ll, num_levels=self.corr_levels, radius=self.corr_radius
        )
        feature_correlation_t_minus_4_func = CorrelationBlock(
            feature_map_ll, num_levels=self.corr_levels, radius=self.corr_radius
        )

        # Template frame
        track_idx = 0

        coords_init = coords_init / spatial_stride

        num_frame, _, feature_h, feature_w = feature_map_ll.shape

        query_features = rearrange(
            bilinear_sample2d(
                feature_map_ll[track_idx], coords_init[:, 0], coords_init[:, 1]
            ),
            "channels num_points -> num_points channels",
        )

        coords = repeat(
            coords_init,
            "num_points num_coords -> num_frame num_points num_coords",
            num_frame=num_frame,
        )
        query_features_ll = repeat(
            query_features,
            "num_points channels -> num_frame num_points channels",
            num_frame=num_frame,
        )

        coord_predictions_ll = []

        feature_correlation_func.corr(query_features_ll)

        query_features_t_minus_2_ll = query_features_ll.clone()
        query_features_t_minus_4_ll = query_features_ll.clone()

        for itr in range(self.num_pips_iter):
            # Comes from here: https://github.com/princeton-vl/RAFT/blob/3fa0bb0a9c633ea0a9bb8a79c576b6785d4e6a02/core/raft.py#L123
            # Improves gradient flow
            # Also see: https://arxiv.org/pdf/1912.10739
            coords = coords.detach()

            if itr >= 1:
                # Multiple template usage
                # timestep indices
                t_minus_2_ll = (torch.arange(num_frame) - 2).clip(min=0)
                t_minus_4_ll = (torch.arange(num_frame) - 4).clip(min=0)

                query_features_t_minus_2_ll = rearrange(
                    bilinear_sample2d(
                        feature_map_ll[t_minus_2_ll],
                        coords[t_minus_2_ll, :, 0],
                        coords[t_minus_2_ll, :, 1],
                    ),
                    "num_frame channels num_points -> num_frame num_points channels",
                )
                query_features_t_minus_4_ll = rearrange(
                    bilinear_sample2d(
                        feature_map_ll[t_minus_4_ll],
                        coords[t_minus_4_ll, :, 0],
                        coords[t_minus_4_ll, :, 1],
                    ),
                    "num_frame channels num_points -> num_frame num_points channels",
                )

            feature_correlation_t_minus_2_func.corr(query_features_t_minus_2_ll)
            feature_correlation_t_minus_4_func.corr(query_features_t_minus_4_ll)

            # now we want costs at the current locations
            feature_correlations: Float[
                Tensor, "num_frame num_points levels_patch_sq"
            ] = feature_correlation_func.sample(coords)
            feature_correlations_t_minus_2: Float[
                Tensor, "num_frame num_points levels_patch_sq"
            ] = feature_correlation_t_minus_2_func.sample(coords)
            feature_correlations_t_minus_4: Float[
                Tensor, "num_frame num_points levels_patch_sq"
            ] = feature_correlation_t_minus_4_func.sample(coords)

            feature_correlations_stacked = torch.cat(
                (
                    feature_correlations,
                    feature_correlations_t_minus_2,
                    feature_correlations_t_minus_4,
                ),
                dim=-1,
            )

            # We are focussing on the offsets---leads to invariant predictions
            # See: https://github.com/aharley/pips2/blob/8b5bd9ecb27274f76f75fcaeff0dbdf13de0b977/nets/pips2.py#L508C23-L508C51
            flow: Float[Tensor, "num_frame_minus_one num_points coords"] = (
                coords[1:] - coords[:-1]
            )
            flow: Float[Tensor, "num_frame num_points coords"] = torch.cat(
                [flow, flow[-1:]], dim=0
            )

            # Float[Tensor, "num_frame num_points coords"]
            delta_coords = self.delta_block(feature_correlations_stacked, flow)

            if beautify and itr > 3 * self.num_pips_iter // 4:
                # this smooths the results a bit, but does not really help perf
                delta_coords = delta_coords * 0.3

            coords = coords + delta_coords
            coords[track_idx] = coords_init  # Coord at first frame
            coord_predictions_ll.append(coords * spatial_stride)

        coord_predictions_ll: Float[
            Tensor, "num_preds num_frame num_points num_coords"
        ] = torch.stack(coord_predictions_ll, dim=0)

        return coord_predictions_ll

    def forward(
        self,
        photon_cube: Bool[Tensor, "h w t"],
        coords_init: Float[Tensor, "num_points num_coords"],
        bocpd_gamma: float = 1e-4,
        hot_pixel_mask: Bool[np.ndarray, "h w"] = None,
        online: bool = False,
        t_init: int = None,
        chaining_length: int = None,
        beautify: bool = False,
        **integrator_kwargs,
    ) -> tuple[
        Float[Tensor, "num_preds num_frame num_points num_coords"],
        Float[Tensor, "..."],
        list[int],
    ]:
        """
        Predicts point tracks for input video (photon_cube) and initial coordinates.

        :param photon_cube: Input tensor, Bool[Tensor, "height width time"]
        :param coords_init: Initial point coords, Float[Tensor, "num_points num_coords"]
        :param bocpd_gamma: (Optional) Integrator gamma
        :return: Tuple (track_predictions, features, time_indices)
        """
        assert coords_init.shape[-1] == 2

        height, width, _ = photon_cube.shape

        # num_frame, channels, height//8, width//8
        if online:
            feature_map_ll, t_index_ll = self.fnet.forward_online(
                photon_cube,
                bocpd_gamma=bocpd_gamma,
                hot_pixel_mask=hot_pixel_mask,
                **integrator_kwargs,
            )
        else:
            feature_map_ll, t_index_ll = self.fnet.forward(
                photon_cube,
                bocpd_gamma=bocpd_gamma,
                hot_pixel_mask=hot_pixel_mask,
                **integrator_kwargs,
            )

        num_frame, _, feature_h, feature_w = feature_map_ll.shape

        # For causality reasons, we want to skip a certain number of initial features
        if t_init is None:
            frame_init = num_frame // 2
        else:
            frame_init = t_init // self.fnet.subsampling

        if chaining_length is None:
            chaining_length = num_frame - frame_init

        spatial_stride_h = height / feature_h
        spatial_stride_w = width / feature_w
        assert spatial_stride_h == spatial_stride_w, "Unequal height and width strides"
        spatial_stride = spatial_stride_h

        # Containers to keep track of what is actually inferred upon
        output_predictions_ll = []

        for frame_idx in range(frame_init, num_frame, chaining_length):
            # Find features corresponding to initialization points

            coord_predictions_ll = self.pred_tracks(
                feature_map_ll[frame_idx : frame_idx + chaining_length],
                coords_init,
                spatial_stride,
                beautify=beautify,
            )
            coords_init = coord_predictions_ll[-1, -1]

            output_predictions_ll.append(coord_predictions_ll)

        output_predictions_ll = torch.cat(output_predictions_ll, dim=1)

        return (
            output_predictions_ll,
            feature_map_ll[frame_init:],
            t_index_ll[frame_init:],
        )


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    h, t = 256, 2048

    photon_cube = torch.rand(h, h, t, device=device)
    logger.info(f"Photon cube of shape {photon_cube.shape}")
    model = PointTracker(subsampling=64).to(device)

    num_points = 16
    bocpd_gamma = 3e-4
    coords_init = torch.zeros(num_points, 2).to(device)

    (
        coord_predictions_ll,
        feature_map_ll,
        t_index_ll,
    ) = model.forward(photon_cube, coords_init, bocpd_gamma=bocpd_gamma)
    logger.info(
        f"Coord predictions.shape (num_frame, num_points, coords): {coord_predictions_ll[-1].shape}"
    )
    coord_predictions_ll[-1].sum().backward()
    logger.info(f"t_index_ll {t_index_ll}")
