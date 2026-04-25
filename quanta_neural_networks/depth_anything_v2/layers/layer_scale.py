"""
Layer scaling implementation for vision transformers.

This module implements learnable layer scaling, which helps with training
stability in deep transformer networks by scaling residual connections.
"""

from typing import Union

import torch
from torch import Tensor
from torch import nn
from jaxtyping import Float


class LayerScale(nn.Module):
    """
    Learnable layer scaling for transformer blocks.
    
    This module applies learnable scaling to input tensors, which helps with
    training stability in deep networks by scaling residual connections.
    """
    
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        """
        Initialize the layer scaling module.
        
        :param dim: Dimension of the scaling parameter
        :param init_values: Initial value(s) for the scaling parameter
        :param inplace: Whether to apply scaling in-place
        """
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Float[Tensor, "batch tokens dim"]) -> Float[Tensor, "batch tokens dim"]:
        """
        Apply layer scaling to the input tensor.
        
        :param x: Input tensor of shape (batch, tokens, dim)
        :return: Scaled tensor of same shape as input
        """
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
