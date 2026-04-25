"""
Transformer block implementation for vision transformers.

This module combines multi-head self-attention with MLP layers to form
a complete transformer block with layer normalization and residual connections.
"""

from typing import Callable

from torch import nn, Tensor
from jaxtyping import Float

from quanta_neural_networks.depth_anything_v2.layers.layer_scale import LayerScale
from quanta_neural_networks.depth_anything_v2.layers.mlp import Mlp
from quanta_neural_networks.depth_anything_v2.layers.attention import Attention


class Block(nn.Module):
    """
    Complete transformer block with attention and MLP layers.
    
    This block implements the standard transformer architecture with:
    - Layer normalization before attention and MLP
    - Multi-head self-attention
    - MLP feed-forward network
    - Residual connections with optional layer scaling
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        init_values: float | None = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
    ) -> None:
        """
        Initialize the transformer block.
        
        :param dim: Token embedding dimension
        :param num_heads: Number of attention heads
        :param mlp_ratio: Ratio of MLP hidden dimension to input dimension
        :param qkv_bias: Whether to use bias in QKV projections
        :param proj_bias: Whether to use bias in attention output projection
        :param ffn_bias: Whether to use bias in MLP layers
        :param drop: Dropout rate
        :param init_values: Initial value for layer scaling (None to disable)
        :param act_layer: Activation layer class
        :param norm_layer: Normalization layer class
        :param attn_class: Attention layer class
        :param ffn_layer: Feed-forward layer class
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )

    def forward(self, x: Float[Tensor, "batch tokens dim"]) -> Float[Tensor, "batch tokens dim"]:
        """
        Apply the transformer block forward pass.
        
        :param x: Input token tensor of shape (batch, tokens, dim)
        :return: Output tensor of same shape as input
        """
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x
