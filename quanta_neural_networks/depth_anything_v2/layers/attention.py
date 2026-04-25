"""
Multi-head self-attention layer for vision transformers.

This module implements the standard multi-head self-attention mechanism used in
vision transformers, closely following the DINOv2 implementation.
"""

from math import sqrt

from torch import nn
from jaxtyping import Float
from torch import Tensor

from quanta_neural_networks.depth_anything_v2.layers.position_encoding import DropPath

LN_EPS = 1e-6


class Attention(nn.Module):
    """
    Multi-head self-attention layer for vision transformers.
    
    This module implements the standard multi-head self-attention mechanism
    used in vision transformers. It computes attention weights between all
    token pairs and applies weighted aggregation of values.
    
    Reference: Based on DINOv2 attention implementation
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        drop_path_rate: float = 0.0,
    ) -> None:
        """
        Initialize the attention layer.
        
        :param dim: The dimensionality of input tokens
        :param num_heads: The number of attention heads
        :param qkv_bias: Whether to use bias in QKV projection
        :param proj_bias: Whether to use bias in output projection
        :param drop_path_rate: Drop path ratio for regularization during training
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        assert 0.0 <= drop_path_rate <= 1.0

        self.scale = sqrt(dim // num_heads)

        self.qkv = nn.Linear(in_features=dim, out_features=dim * 3, bias=qkv_bias)
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

        self.proj = nn.Linear(in_features=dim, out_features=dim, bias=proj_bias)

    def forward(self, x: Float[Tensor, "batch tokens dim"]) -> Float[Tensor, "batch tokens dim"]:
        """
        Apply multi-head self-attention to input tokens.
        
        :param x: Input token tensor of shape (batch, tokens, dim)
        :return: Output tensor of same shape as input
        """
        # Linearly project x into qkv space.
        x = self.qkv(x)

        # Compute attention on the qkv representation.
        # (batch, token, dim)
        # Partition the windows and attention heads. Windows are arranged
        # along the batch dimension.
        q, k, v = self._partition_heads(x)
        # (batch, heads, token, dim / heads)

        # Perform the actual attention computation.
        # The output of this first matmul is huge - hence it's much
        # faster to scale one of the inputs than it is to scale the
        # output.
        x = (q / self.scale) @ k.transpose(-2, -1)
        x = x.softmax(dim=-1) @ v
        # (batch, heads, token, dim / heads)

        x = self._recombine_heads(x)
        # (batch, token, dim)

        # Apply the post-attention linear transform and add the skip.
        x = self.proj(x)
        x = self.drop_path(x)

        return x

    def _partition_heads(self, x: Float[Tensor, "batch tokens dim"]) -> tuple[Float[Tensor, "batch heads tokens dim_per_head"], ...]:
        """
        Partition the QKV tensor into separate query, key, and value tensors.
        
        :param x: QKV tensor of shape (batch, tokens, dim*3)
        :return: Tuple of (q, k, v) tensors each of shape (batch, heads, tokens, dim/heads)
        """
        # (batch, token, dim)

        x = x.view(
            x.shape[:-1] + (3, self.num_heads, x.shape[-1] // (3 * self.num_heads))
        )
        q, k, v = x.permute(2, 0, 3, 1, 4)
        # (batch, heads, token, dim / heads)

        return q, k, v

    @staticmethod
    def _recombine_heads(x: Float[Tensor, "batch heads tokens dim_per_head"]) -> Float[Tensor, "batch tokens dim"]:
        """
        Recombine multi-head attention output back to original token dimension.
        
        :param x: Multi-head tensor of shape (batch, heads, tokens, dim/heads)
        :return: Recombined tensor of shape (batch, tokens, dim)
        """
        # (batch, heads, token, dim / heads)

        # Can't use x.view here because of the permutation.
        x = x.permute(0, 2, 1, 3)
        x_reshaped = x.reshape(x.shape[:-2] + (-1,))
        # (batch, token, dim)

        assert x.data_ptr() != x_reshaped.data_ptr()
        x = x_reshaped

        return x
