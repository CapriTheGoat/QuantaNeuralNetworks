# This file contains code adapted from Meta's DINO project (https://github.com/facebookresearch/dino),
# licensed under the Apache License, Version 2.0 (the "License").
# See LICENSE or http://www.apache.org/licenses/LICENSE-2.0 for details.
#
# Additional modifications are subject to the primary license of this project.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import math
from functools import partial
from typing import Sequence, Tuple, Union, Callable

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
from einops import rearrange
from jaxtyping import Bool, Float
from loguru import logger
from torch import Tensor
from torch.nn.init import trunc_normal_
from tqdm import tqdm

from quanta_neural_networks.depth_anything_v2.layers.block import Block
from quanta_neural_networks.depth_anything_v2.layers.mlp import Mlp
from quanta_neural_networks.depth_anything_v2.layers.patch_embed import PatchEmbed
from quanta_neural_networks.depth_anything_v2.transform import Resize
from quanta_neural_networks.integrator import PerPixelBayesian
from quanta_neural_networks.ssd import SSD
from quanta_neural_networks.utils.train_utils import freeze_module


def named_apply(
    fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class DinoVisionTransformer(nn.Module):
    """
    Transformer backbone adapted from Meta's DINO (https://github.com/facebookresearch/dino)
    under Apache License 2.0. See file header for license details.

    :param dino_img_size: Input image size
    :param vit_patch_size: Patch size
    :param patch_stride_factor: Fraction of stride vs patch size, 0..1
    :param in_chans: Channels in input image
    :param embed_dim: Embedding dimension
    :param depth: Number of transformer layers
    :param num_heads: Attention heads
    :param mlp_ratio: Ratio of MLP hidden dim to embedding dim
    :param qkv_bias: Use bias in QKV projections
    :param proj_bias: Use bias in projection
    :param ffn_bias: Use bias in FFN
    :param init_values: Initial value for layer scaling
    :param act_layer: Activation layer (default: GELU)
    :param num_register_tokens: Extra register tokens (DINOv2)
    :param interpolate_antialias: Apply antialias for embedding resizing
    :param interpolate_offset: Positional embedding interpolation offset

    .. note::
        This architecture is directly adapted from Meta's DINO and DINOv2 projects, Apache-2.0.
    """

    def __init__(
        self,
        dino_img_size=518,
        vit_patch_size=16,
        patch_stride_factor: float = 1,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        act_layer=nn.GELU,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    ):
        """
        Args:
            dino_img_size (int, tuple): input image size
            vit_patch_size (int, tuple): patch size
            patch_stride_factor (float): how much to stride by during patch embedding.
                0...1, as a fraction of the patch size.
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            init_values (float): layer-scale init values
            act_layer (nn.Module): MLP activation layer
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating positional embeddings
            interpolate_offset: (float) work-around offset to apply when interpolating positional embeddings
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other integrator
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.vit_patch_size = vit_patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset
        self.depth = depth
        self.dino_img_size = dino_img_size
        self.patch_stride_factor = patch_stride_factor

        self.patch_embed = PatchEmbed(
            img_size=dino_img_size,
            patch_size=vit_patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            stride_factor=patch_stride_factor,
        )
        num_patches = self.patch_embed.num_patches_unstrided

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_tokens, embed_dim)
        )
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
            if num_register_tokens
            else None
        )

        blocks_list = [
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=Mlp,
                init_values=init_values,
            )
            for _ in range(depth)
        ]

        self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(
        self, x: Float[Tensor, "batch seq embed_dim"], w: int, h: int
    ) -> Float[Tensor, "batch seq embed_dim"]:
        """
        Interpolates positional encoding for variable-sized inputs.

        :param x: Image tokens, Float[Tensor, "batch seq embed_dim"]
        :param w: Width after patchifying
        :param h: Height after patchifying
        :return: Adjusted positional embeddings, Float[Tensor, "batch seq embed_dim"]
        """
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]

        # See: https://github.com/AssafSinger94/dino-tracker/blob/7f368958a1c69e31f67f48c410f2cad3f7be8402/models/extractor.py#L70
        stride = int(self.vit_patch_size * self.patch_stride_factor)
        w0 = 1 + (w - self.vit_patch_size) // stride
        h0 = 1 + (h - self.vit_patch_size) // stride
        assert (
            w0 * h0 == npatch
        ), f"""got wrong grid size for {h}x{w} with patch_size {self.vit_patch_size} and 
        stride {stride} got {h0}x{w0}={h0 * w0} expecting {npatch}"""

        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        # DINOv2 with register modify the interpolate_offset from 0.1 to 0.0
        w0, h0 = w0 + self.interpolate_offset, h0 + self.interpolate_offset

        sqrt_N = math.sqrt(N)
        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(
                0, 3, 1, 2
            ),
            scale_factor=(sx, sy),
            # (int(w0), int(h0)), # to solve the upsampling shape issue
            mode="bicubic",
            antialias=self.interpolate_antialias,
        )

        assert int(w0) == patch_pos_embed.shape[-2]
        assert int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(
            previous_dtype
        )

    def prepare_tokens_with_masks(
        self,
        x: Float[Tensor, "batch channels height width"],
        masks: Bool[Tensor, "batch num_patches"] = None,
    ) -> Float[Tensor, "batch seq embed_dim"]:
        """
        Prepares input tokens by adding class tokens, processing masks, and encoding position.

        :param x: Image input, Float[Tensor, "batch channels height width"]
        :param masks: Optional masks, Bool[Tensor, "batch num_patches"]
        :return: Token tensor, Float[Tensor, "batch seq embed_dim"]
        """
        B, nc, w, h = x.shape
        x = self.patch_embed(x)

        if masks is not None:
            x = torch.where(
                masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x
            )

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        return x

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = (
            range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        )
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)

        assert len(output) == len(
            blocks_to_take
        ), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: Float[Tensor, "batch channels height width"],
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm: bool = True,
    ) -> Tuple[Float[Tensor, "batch ..."], ...]:
        """
        Returns intermediate layer outputs.

        :param x: Input batch, Float[Tensor, "batch channels height width"]
        :param n: Number/layers to extract
        :param reshape: Reshape to feature map layout
        :param return_class_token: Also return class token output
        :param norm: Apply normalization
        :return: Tuple of outputs per selected layer
        """
        outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens :] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(
                    B,
                    self.patch_embed.get_width_patch_num(w),
                    self.patch_embed.get_height_patch_num(h),
                    -1,
                )
                .permute(0, 3, 1, 2)
                .contiguous()
                for out in outputs
            ]

        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args):
        raise NotImplementedError


class DinoVisionTransformerSSD(DinoVisionTransformer):
    """
    Dino backbone with SSD head, adapted from Meta DINO and DepthAnything (https://github.com/isl-org/Depth-Anything)
    under Apache License 2.0 and original DepthAnything license.
    For license details see repo links and header.

    :param state_dim: State vector dimension used in SSD
    :param subsampling: Frame/time subsampling factor
    :param memory_size: Memory window for integrator
    Other parameters as in DinoVisionTransformer.

    .. note::
      Derived from Meta DINO (Apache-2.0) with DepthAnything-v2 design conventions.
    """

    def __init__(
        self,
        state_dim: int = 8,
        subsampling: int = 64,
        memory_size: int = 20,
        freeze_spatial: bool = True,
        dino_img_size=518,
        vit_patch_size=16,
        patch_stride_factor: float = 1,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        init_values=None,
        act_layer=nn.GELU,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        **integrator_kwargs,
    ):
        super().__init__(
            dino_img_size=dino_img_size,
            vit_patch_size=vit_patch_size,
            patch_stride_factor=patch_stride_factor,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            ffn_bias=ffn_bias,
            proj_bias=proj_bias,
            init_values=init_values,  # for layerscale: None or 0 => no layerscale
            act_layer=act_layer,
            num_register_tokens=num_register_tokens,
            interpolate_antialias=interpolate_antialias,
            interpolate_offset=interpolate_offset,
        )

        self.subsampling = subsampling

        self.integrator = PerPixelBayesian(
            **integrator_kwargs, normalize=True, subsampling=subsampling
        )
        logger.info(f"Using integrator {self.integrator} with Dino-v2-photons")

        self.memory_size = memory_size

        self.ssm_patch = SSD(
            subsampling=self.subsampling,
            state_dim=state_dim,
            in_dim=self.embed_dim,
            head_dim=self.embed_dim // 3,
            identity_init=True,
        )

        self.ssm_ll = nn.ModuleList()

        for ssm_idx in range(self.depth):
            ssm_block = SSD(
                in_dim=self.embed_dim,
                head_dim=self.embed_dim // 3,
                state_dim=state_dim,
                identity_init=True,
            )

            self.ssm_ll.append(ssm_block)

        self.mean = nn.Parameter(
            torch.tensor([0.485, 0.456, 0.406]), requires_grad=False
        )
        self.std = nn.Parameter(
            torch.tensor([0.229, 0.224, 0.225]), requires_grad=False
        )

        self.resize = Resize(
            width=dino_img_size,
            height=dino_img_size,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        )

        # Freeze portions of the model
        if freeze_spatial:
            for blk in self.blocks:
                freeze_module(blk)
            freeze_module(self.patch_embed)
            freeze_module(self.norm)
            self.mask_token.requires_grad = False
            self.cls_token.requires_grad = False
            self.pos_embed.requires_grad = False

        # Submodules where we want to capture layer stats

        self.t_prev = 0

    def _get_intermediate_layers_not_chunked(
        self,
        photon_cube: Bool[Tensor, "h w t"],
        bocpd_gamma: float = 1e-4,
        min_window: int = 32,
        n=1,
        quantile: float = 1.0,
        hot_pixel_mask: Bool[np.ndarray, "h w"] = None,
        **integrator_kwargs,
    ) -> tuple[Float[Tensor, "seq_len batch channels"], tuple[int, int], list[int]]:
        h, w, t = photon_cube.shape
        t_index_ll = list(range(1, t + 1))

        with torch.no_grad():
            x = self.integrator.process_photon_cube(
                photon_cube,
                bocpd_gamma=bocpd_gamma,
                min_window=min_window,
                hot_pixel_mask=hot_pixel_mask,
                subsampling=self.subsampling,
                quantile=quantile,
                **integrator_kwargs,
            )

            t_index_ll = t_index_ll[self.subsampling - 1 :: self.subsampling]
            x = rearrange(x, "h w t -> t 1 h w")
            x = self.resize.forward_tensor(x)

            h, w = x.shape[-2:]

            # Repeat and normalize across channel dim
            x = (x - self.mean.reshape(1, -1, 1, 1)) / self.std.reshape(1, -1, 1, 1)

        x = self.prepare_tokens_with_masks(x)
        x, t_index_ll = self.ssm_patch(x, t_index_ll)

        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = (
            range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        )

        for e, blk in enumerate(self.blocks):
            x = blk(x)
            x, t_index_ll = self.ssm_ll[e](x, t_index_ll)

            if e in blocks_to_take:
                output.append(x)

            if e >= max(blocks_to_take):
                break

        assert len(output) == len(
            blocks_to_take
        ), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output, (h, w), t_index_ll

    def clear_hidden_state(self):
        self.t_prev = 0
        self.ssm_patch.clear_hidden_state()
        for ssm in self.ssm_ll:
            ssm.clear_hidden_state()

    def reset(self, recurse=True):
        super().reset(recurse)
        self.clear_hidden_state()

    def _get_intermediate_layers_not_chunked_online(
        self,
        photon_cube: Bool[Tensor, "h w t"],
        bocpd_gamma: float = 1e-4,
        min_window: int = 32,
        n=1,
        quantile: float = 1.0,
        clear_states: bool = True,
        hot_pixel_mask: Bool[np.ndarray, "h w"] = None,
        use_tqdm: bool = False,
        **integrator_kwargs,
    ) -> tuple[
        list[Float[Tensor, "seq_len batch channels"]], tuple[int, int], list[int]
    ]:
        h, w, t = photon_cube.shape

        if clear_states:
            self.clear_hidden_state()
        t_index_ll = np.arange(1 + self.t_prev, t + 1 + self.t_prev)

        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = (
            range(len(self.blocks) - n, len(self.blocks)) if isinstance(n, int) else n
        )
        output_dict = {e: [] for e in blocks_to_take}

        recons_ll = self.integrator.process_photon_cube(
            photon_cube,
            bocpd_gamma=bocpd_gamma,
            min_window=min_window,
            hot_pixel_mask=hot_pixel_mask,
            subsampling=self.subsampling,
            quantile=quantile,
            clear_states=clear_states,
            **integrator_kwargs,
        )
        recons_ll = rearrange(recons_ll, "h w t -> t 1 h w")
        t_index_ll = t_index_ll[self.subsampling - 1 :: self.subsampling]

        out_t_index_ll = []
        recons_ll = self.resize.forward_tensor(recons_ll)
        h, w = recons_ll.shape[-2:]

        pbar = t_index_ll
        if use_tqdm:
            pbar = tqdm(t_index_ll)

        for recons_idx, t_index in enumerate(pbar):
            x = recons_ll[recons_idx]

            if not t_index % self.subsampling == 0:
                continue

            # Repeat and normalize across channel dim
            x = (x - self.mean.reshape(-1, 1, 1)) / self.std.reshape(-1, 1, 1)

            x = self.prepare_tokens_with_masks(x.unsqueeze(0)).squeeze(0)
            x = self.ssm_patch.forward_online(x, t_index)

            for e, blk in enumerate(self.blocks):
                x = blk(x.unsqueeze(0)).squeeze(0)
                x: Float[Tensor, "batch channels"] = self.ssm_ll[e].forward_online(
                    x, t_index
                )

                if e in blocks_to_take:
                    output_dict[e].append(x)

                if e >= max(blocks_to_take):
                    break

            out_t_index_ll.append(t_index)

        for e in output_dict:
            stride = len(output_dict[e]) // len(output_dict[blocks_to_take[-1]])
            output_dict[e] = output_dict[e][stride - 1 :: stride]
            output_dict[e] = torch.stack(output_dict[e], dim=0)

        self.t_prev += t
        output = output_dict.values()
        assert len(output) == len(
            blocks_to_take
        ), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output, (h, w), out_t_index_ll

    def get_intermediate_layers(
        self,
        photon_cube: Bool[Tensor, "h w t"],
        bocpd_gamma: float = 1e-4,
        min_window: int = 32,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm: bool = True,
        online: bool = False,
        quantile: float = 1.0,
        hot_pixel_mask: Bool[np.ndarray, "h w"] = None,
        use_tqdm: bool = False,
        clear_states: bool = True,
        **integrator_kwargs,
    ) -> tuple[Tuple[Float[Tensor, "..."], ...], list[int]]:
        if online:
            (
                outputs,
                (h, w),
                t_index_ll,
            ) = self._get_intermediate_layers_not_chunked_online(
                photon_cube,
                bocpd_gamma=bocpd_gamma,
                min_window=min_window,
                n=n,
                quantile=quantile,
                hot_pixel_mask=hot_pixel_mask,
                reshape_stats=reshape,
                use_tqdm=use_tqdm,
                clear_states=clear_states,
                **integrator_kwargs,
            )
        else:
            outputs, (h, w), t_index_ll = self._get_intermediate_layers_not_chunked(
                photon_cube,
                bocpd_gamma=bocpd_gamma,
                min_window=min_window,
                n=n,
                quantile=quantile,
                hot_pixel_mask=hot_pixel_mask,
                **integrator_kwargs,
            )

        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens :] for out in outputs]

        if reshape:
            h_output = self.patch_embed.get_height_patch_num(h)
            w_output = self.patch_embed.get_width_patch_num(w)

            outputs = [
                rearrange(
                    out,
                    "seq_len (h w) channels -> seq_len channels h w",
                    h=h_output,
                    w=w_output,
                ).contiguous()
                for out in outputs
            ]

        if return_class_token:
            return tuple(zip(outputs, class_tokens)), t_index_ll
        return tuple(outputs), t_index_ll


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        if not module.weight.is_complex():
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


# See: https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md
dino_model_args = {
    "vits": dict(
        dino_img_size=518,
        vit_patch_size=14,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
    ),
    "vitb": dict(
        dino_img_size=518,
        vit_patch_size=14,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
    ),
    "vitl": dict(
        dino_img_size=518,
        vit_patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
    ),
}

# See: https://github.com/DepthAnything/Depth-Anything-V2/blob/31dc97708961675ce6b3a8d8ffa729170a4aa273/depth_anything_v2/dinov2.py#L406
depth_anything_v2_model_args = {
    k: {
        **dino_model_args[k],
        **dict(
            init_values=1.0,
        ),
    }
    for k in dino_model_args
}
