"""
Patch embedding implementation for vision transformers.

This module converts 2D images into patch embeddings by dividing the image
into patches and projecting each patch to a higher-dimensional space.
"""

from typing import Callable, Optional, Tuple, Union

import torch.nn as nn
from torch import Tensor
from jaxtyping import Float


def make_2tuple(x: int | tuple[int, int]) -> tuple[int, int]:
    """
    Convert input to a 2-tuple.
    
    :param x: Input value (int or tuple of length 2)
    :return: Tuple of length 2
    """
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)
    
    This module converts input images into patch embeddings by dividing
    the image into non-overlapping patches and projecting each patch
    to a higher-dimensional embedding space.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        stride_factor: float = 1.0,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ) -> None:
        """
        Initialize the patch embedding module.
        
        :param img_size: Input image size
        :param patch_size: Size of each patch
        :param stride_factor: Stride factor for patch extraction
        :param in_chans: Number of input channels
        :param embed_dim: Embedding dimension
        :param norm_layer: Normalization layer
        :param flatten_embedding: Whether to flatten embeddings
        """
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        self.patch_size = patch_size
        self.stride = int(patch_size * stride_factor)

        stride_HW = make_2tuple(self.stride)
        patch_grid_size = (
            self.get_height_patch_num(image_HW[0]),
            self.get_width_patch_num(image_HW[1]),
        )

        self.img_size = image_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.num_patches_unstrided = image_HW[0] * image_HW[1] // pow(patch_size, 2)

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_HW,
            stride=stride_HW,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def get_width_patch_num(self, width: int) -> int:
        """
        Calculate number of patches in width dimension.
        
        :param width: Image width
        :return: Number of patches in width dimension
        """
        return 1 + (width - self.patch_size) // self.stride

    def get_height_patch_num(self, height: int) -> int:
        """
        Calculate number of patches in height dimension.
        
        :param height: Image height
        :return: Number of patches in height dimension
        """
        return 1 + (height - self.patch_size) // self.stride

    def get_patch_num(self, image_shape: tuple[int, int]) -> int:
        """
        Calculate total number of patches for given image shape.
        
        :param image_shape: (height, width) tuple
        :return: Total number of patches
        """
        height, width = image_shape
        return 1 + (self.get_height_patch_num(height) * self.get_width_patch_num(width))

    def forward(self, x: Float[Tensor, "batch channels height width"]) -> Float[Tensor, "batch num_patches embed_dim"]:
        """
        Convert input image to patch embeddings.
        
        :param x: Input image tensor of shape (batch, channels, height, width)
        :return: Patch embeddings of shape (batch, num_patches, embed_dim)
        """
        _, _, H, W = x.shape
        patch_H = patch_W = self.patch_size

        assert (
            H % patch_H == 0
        ), f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert (
            W % patch_W == 0
        ), f"Input image width {W} is not a multiple of patch width: {patch_W}"
        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x
