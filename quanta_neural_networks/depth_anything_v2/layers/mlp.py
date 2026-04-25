"""
Multi-layer perceptron (MLP) implementation for vision transformers.

This module implements the feed-forward network component of transformer blocks,
typically consisting of two linear layers with an activation function and dropout.
"""

from typing import Callable, Optional

from torch import Tensor, nn
from jaxtyping import Float


class Mlp(nn.Module):
    """
    Multi-layer perceptron (MLP) for transformer blocks.
    
    This module implements the standard MLP used in transformer architectures,
    consisting of two linear layers with an activation function between them.
    """
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        """
        Initialize the MLP layer.
        
        :param in_features: Input feature dimension
        :param hidden_features: Hidden layer dimension (defaults to in_features)
        :param out_features: Output feature dimension (defaults to in_features)
        :param act_layer: Activation layer class
        :param drop: Dropout rate
        :param bias: Whether to use bias in linear layers
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(
            in_features,
            hidden_features,
            bias=bias,
        )
        self.act = act_layer()
        self.fc2 = nn.Linear(
            hidden_features,
            out_features,
            bias=bias,
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x: Float[Tensor, "batch tokens dim"]) -> Float[Tensor, "batch tokens dim"]:
        """
        Apply the MLP forward pass.
        
        :param x: Input tensor of shape (batch, tokens, dim)
        :return: Output tensor of same shape as input
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)

        x = self.drop(x)
        return x
