"""
Correlation computation for point tracking.

This module implements correlation-based matching for point tracking,
including bilinear sampling and correlation pyramid construction.
"""

from math import sqrt

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor
from torch.nn import functional as F


def bilinear_sampler(
    img: Float[Tensor, "batch channels height width"],
    coords: Float[Tensor, "batch coords height width"],
    return_mask: bool = False,
) -> Float[Tensor, "batch channels height width"] | tuple[Float[Tensor, "batch channels height width"], Float[Tensor, "batch height width"]]:
    """
    Bilinear sampling wrapper for grid_sample using pixel coordinates.
    
    :param img: Input image tensor of shape (batch, channels, height, width)
    :param coords: Coordinate tensor of shape (batch, coords, height, width)
    :param return_mask: Whether to return validity mask
    :return: Sampled image tensor, optionally with validity mask
    """
    height, width = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    # go to 0,1 then 0,2 then -1,1
    xgrid = 2 * xgrid / (width - 1) - 1
    ygrid = 2 * ygrid / (height - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    # grid values between -1...1
    if return_mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()
    return img


class CorrelationBlock:
    """
    Correlation block for multi-scale feature matching.
    
    This class builds a correlation pyramid from feature maps and provides
    methods to sample correlations at different scales for point tracking.
    """
    
    def __init__(self, feature_maps: Float[Tensor, "num_frame channels height width"], num_levels: int = 4, radius: int = 4) -> None:
        """
        Initialize the correlation block.
        
        :param feature_maps: Feature maps of shape (num_frame, channels, height, width)
        :param num_levels: Number of pyramid levels
        :param radius: Correlation radius for sampling
        """
        num_frame, channels, height, width = feature_maps.shape
        self.num_frame, self.channels, self.height, self.width = (
            num_frame,
            channels,
            height,
            width,
        )
        self.num_levels = num_levels
        self.radius = radius
        self.feature_maps_pyramid = []
        self.feature_maps_pyramid.append(feature_maps)
        for i in range(self.num_levels - 1):
            feature_maps = F.avg_pool2d(feature_maps, 2, stride=2)
            self.feature_maps_pyramid.append(feature_maps)

    def sample(
        self, coords: Float[Tensor, "num_frame num_points num_coords"]
    ) -> Float[Tensor, "num_frame num_points levels_patch_sq"]:
        """
        Sample correlations at given coordinates across pyramid levels.
        
        :param coords: Coordinate tensor of shape (num_frame, num_points, num_coords)
        :return: Sampled correlations of shape (num_frame, num_points, levels_patch_sq)
        """
        num_frame, num_points, num_coords = coords.shape
        assert num_coords == 2

        out_pyramid = []
        for i in range(self.num_levels):
            corrs = self.correlations_pyramid[i]
            _, _, height, width = corrs.shape

            dx = torch.linspace(-self.radius, self.radius, 2 * self.radius + 1)
            dy = torch.linspace(-self.radius, self.radius, 2 * self.radius + 1)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), dim=-1).to(
                coords.device
            )

            centroid_lvl = coords.reshape(num_frame * num_points, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2 * self.radius + 1, 2 * self.radius + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corrs = bilinear_sampler(
                corrs.reshape(num_frame * num_points, 1, height, width),
                coords_lvl,
            )
            corrs = corrs.view(num_frame, num_points, -1)
            out_pyramid.append(corrs)

        out: Float[Tensor, "num_frame num_points levels_patch_sq"] = torch.cat(
            out_pyramid, dim=-1
        )  # Num frame, LRR*2
        return out

    def get_cost_volume(
        self, query_features: Float[Tensor, "num_points channels"]
    ) -> Float[Tensor, "num_frame num_points h w"]:
        """
        Compute cost volume from query features.
        
        :param query_features: Query features of shape (num_points, channels)
        :return: Cost volume of shape (num_frame, num_points, h, w)
        """
        num_points, channels = query_features.shape
        assert channels == self.channels

        feature_maps = self.feature_maps_pyramid[0]
        _, _, height, width = feature_maps.shape

        # s: num_frame, n: num_points, c: channels
        correlations = torch.einsum("nc,schw->snhw", query_features, feature_maps)

        # Normalize
        query_norm = rearrange(
            query_features.norm(dim=1), "num_points -> 1 num_points 1 1"
        )
        feature_maps_norm = rearrange(
            feature_maps.norm(dim=1), "num_frame h w -> num_frame 1 h w"
        )

        # Cosine similarity
        cost_volume = correlations / (query_norm * feature_maps_norm).clamp(min=1e-6)
        return cost_volume

    def corr(self, targets: Float[Tensor, "num_frame num_points channels"]) -> None:
        """
        Compute correlations between targets and feature maps.
        
        :param targets: Target features of shape (num_frame, num_points, channels)
        """
        num_frame, num_points, channels = targets.shape
        assert channels == self.channels

        self.correlations_pyramid = []
        for feature_maps in self.feature_maps_pyramid:
            _, _, height, width = feature_maps.shape

            # Flatten
            fmap2s = rearrange(
                feature_maps,
                "num_frame channels height width -> num_frame channels (height width)",
            )  # fmaps.view(num_frame, coords, height * width)
            correlations: Float[Tensor, "num_frame num_points hw"] = torch.matmul(
                targets, fmap2s
            )
            correlations = rearrange(
                correlations,
                "num_frame num_points (height width) -> num_frame num_points height width",
                height=height,
                width=width,
            )
            correlations = correlations / sqrt(channels)
            self.correlations_pyramid.append(correlations)
