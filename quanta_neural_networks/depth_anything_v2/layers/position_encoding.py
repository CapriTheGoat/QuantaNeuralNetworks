"""
Position encoding and regularization modules for vision transformers.

This module contains implementations of drop path regularization and various
position encoding strategies used in vision transformer architectures.
"""

from math import prod

import torch
from jaxtyping import Float
from torch import nn as nn, Tensor
from torch.nn import functional as func


class DropPath(nn.Module):
    """
    Drop path (stochastic depth) regularization module.
    
    This module implements stochastic depth regularization by randomly
    dropping entire paths during training, which helps prevent overfitting
    and improves generalization.
    
    Reference: https://github.com/alibaba-mmai-research/TAdaConv/blob/main/models/base/base_blocks.py
    """

    def __init__(self, drop_rate: float) -> None:
        """
        Initialize the drop path module.
        
        :param drop_rate: Fraction of paths to drop (0.0 to 1.0)
        """
        super().__init__()
        self.drop_rate = drop_rate

    def forward(self, x: Float[Tensor, "... batch dim"]) -> Float[Tensor, "... batch dim"]:
        """
        Apply drop path regularization during training.
        
        :param x: Input tensor
        :return: Tensor with dropped paths during training, unchanged during inference
        """
        if not self.training:
            return x
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        keep_mask = torch.rand(shape, device=x.device) > self.drop_rate
        output = x.div(1.0 - self.drop_rate) * keep_mask.to(x.dtype)
        return output

    def __repr__(self) -> str:
        """Return string representation of the module."""
        msg = super().__repr__()
        msg += f"(drop_rate={self.drop_rate})"
        return msg


class PositionEncoding(nn.Module):
    """
    Learnable position encoding for vision transformers.
    
    This module adds learnable position embeddings to input tokens to provide
    spatial information to the transformer architecture.
    """

    def __init__(self, dim: int, encoding_size: tuple[int, ...], input_size: tuple[int, ...], has_class_token: bool) -> None:
        """
        Initialize the position encoding module.
        
        :param dim: The dimensionality of token vectors
        :param encoding_size: The size (in tokens) assumed for position encodings
        :param input_size: The expected size of the inputs in tokens
        :param has_class_token: Whether the input has a class token
        """
        super().__init__()
        self.encoding_size = tuple(encoding_size)
        self.input_size = tuple(input_size)
        self.has_class_token = has_class_token
        tokens = prod(self.encoding_size) + int(has_class_token)
        self.encoding = nn.Parameter(torch.zeros(1, tokens, dim))
        self.cached_encoding = None

    def forward(self, x: Float[Tensor, "batch tokens token_dim"]) -> Float[Tensor, "batch tokens token_dim"]:
        """
        Add position encoding to input tokens.
        
        :param x: Input token tensor of shape (batch, tokens, token_dim)
        :return: Tensor with position encoding added
        """
        if self.training:
            self.cached_encoding = None
            encoding = self._compute_sized_encoding()
        else:
            # Cache the resized encoding during inference (assuming the
            # weights don't change, its value doesn't change between
            # model invocations).
            if self.cached_encoding is None:
                self.cached_encoding = self._compute_sized_encoding()
            encoding = self.cached_encoding

        # Add the position encoding.
        x += encoding
        return x

    def _compute_sized_encoding(self) -> Float[Tensor, "batch tokens dim"]:
        """
        Compute position encoding with appropriate size for current input.
        
        :return: Position encoding tensor of shape (batch, tokens, dim)
        """
        encoding = self.encoding

        # Interpolate the position encoding if needed.
        if self.input_size != self.encoding_size:
            # (batch, patch, dim)

            if self.has_class_token:
                # The class token comes *first* (see ViViTSubModel).
                class_token = encoding[:, :1]
                encoding = encoding[:, 1:]
            else:
                class_token = None
            encoding = encoding.transpose(1, 2)
            encoding = encoding.view(encoding.shape[:-1] + self.encoding_size)
            # (batch, dim) + encoding_size

            # Note: We do not count operations from this interpolation,
            # even though it is in the backbone. This is because the
            # cost of interpolating is amortized over many invocations.
            encoding = func.interpolate(
                encoding, self.input_size, mode="bicubic", align_corners=False
            )
            # (batch, dim) + embedding_size

            encoding = encoding.flatten(start_dim=2)
            encoding = encoding.transpose(1, 2)
            if self.has_class_token:
                encoding = torch.concat([class_token, encoding], dim=1)
            # (batch, patch, dim)

        return torch.Tensor(encoding)

    def reset_self(self) -> None:
        """
        Clear cached position encoding.
        
        Clear the cached value of sized_encoding whenever the model is
        reset (just in case new weights get loaded).
        """
        self.cached_encoding = None


class RelativePositionEmbedding(nn.Module):
    """
    Relative position embedding for attention mechanisms.
    
    This module implements relative position embeddings that encode the relative
    spatial relationships between tokens in attention computations.
    """

    def __init__(self, attention_size: tuple[int, ...], embedding_size: tuple[int, ...], head_dim: int, pool_size: tuple[int, ...] | None = None) -> None:
        """
        Initialize the relative position embedding module.
        
        :param attention_size: The expected size of the attention window
        :param embedding_size: The size (in tokens) assumed for position embeddings
        :param head_dim: The dimensionality of each attention head
        :param pool_size: The pooling size (if self-attention pooling is being used)
        """
        super().__init__()
        self.attention_size = attention_size
        self.embedding_size = embedding_size
        self.pool_size = pool_size
        self.y_embedding = nn.Parameter(
            torch.zeros(2 * embedding_size[0] - 1, head_dim)
        )
        self.x_embedding = nn.Parameter(
            torch.zeros(2 * embedding_size[1] - 1, head_dim)
        )
        self.y_relative = None
        self.x_relative = None

    # This is based on the add_decomposed_rel_pos function here:
    # https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/utils.py
    # noinspection PyTypeChecker
    def forward(self, x: Float[Tensor, "batch heads tokens dim"], q: Float[Tensor, "batch heads tokens dim"]) -> Float[Tensor, "batch heads tokens dim"]:
        """
        Apply relative position embedding to attention scores.
        
        :param x: Attention scores tensor
        :param q: Query tensor
        :return: Attention scores with relative position information added
        """
        a = self.attention_size

        # Unflatten the spatial dimensions.
        if self.pool_size is None:
            p = a
        else:
            p = (a[0] // self.pool_size[0], a[1] // self.pool_size[1])
        x = x.view(x.shape[:2] + a + p)
        q = q.view(q.shape[:2] + a + q.shape[-1:])

        # Apply the relative position embedding.
        if self.y_relative is None:
            # Cache y_relative and x_relative (assuming the weights
            # don't change, their values don't change between model
            # invocations).
            self.y_relative = self._get_relative(self.y_embedding, dim=0)
            self.x_relative = self._get_relative(self.x_embedding, dim=1)
        x += (torch.einsum("abhwc,hkc->abhwk", q, self.y_relative).unsqueeze(dim=-1),)

        x += (torch.einsum("abhwc,wkc->abhwk", q, self.x_relative).unsqueeze(dim=-2),)

        # Re-flatten the spatial dimensions.
        x = x.view(x.shape[:2] + (prod(a), prod(p)))

        return x

    # This is a simplification of the get_rel_pos function here:
    # https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/utils.py
    def _get_relative(self, embedding: Float[Tensor, "embed_size head_dim"], dim: int) -> Float[Tensor, "attention_size attention_size head_dim"]:
        """
        Compute relative position embedding for a given dimension.
        
        :param embedding: Position embedding tensor
        :param dim: Dimension index (0 for y, 1 for x)
        :return: Relative position embedding tensor
        """
        range_0 = torch.arange(self.embedding_size[dim]).unsqueeze(dim=1)
        range_1 = torch.arange(self.embedding_size[dim]).unsqueeze(dim=0)
        relative = embedding[range_0 - range_1 + self.embedding_size[dim] - 1]
        if self.embedding_size != self.attention_size:
            relative = relative.transpose(0, 2).unsqueeze(dim=0)
            relative = func.interpolate(
                relative, self.attention_size, mode="bicubic", align_corners=False
            )
            relative = relative.squeeze(dim=0).transpose(0, 2)
        if self.pool_size is not None:
            relative = relative.transpose(1, 2)
            relative = func.avg_pool1d(relative, self.pool_size[dim])
            relative = relative.transpose(1, 2)
        return relative

    def reset_self(self) -> None:
        """
        Clear cached relative position embeddings.
        
        Clear the cached values of x_relative and y_relative whenever
        the model is reset (just in case new weights get loaded).
        """
        self.y_relative = None
        self.x_relative = None
