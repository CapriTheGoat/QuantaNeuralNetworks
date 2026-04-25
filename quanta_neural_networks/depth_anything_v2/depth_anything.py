"""
This module implements the DepthAnything-v2 models using Meta's DINO backbone (Apache-2.0)
    and design concepts from the DepthAnything-v2 project (https://github.com/isl-org/Depth-Anything).

.. note::
    The DINO backbone (DinoVisionTransformer) is licensed under Apache-2.0 from Meta.
    Additional modification is subject to the primary license of this project.
"""
import cv2
import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float, Bool
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.transforms import Compose

from quanta_neural_networks.depth_anything_v2.dino_v2 import (
    DinoVisionTransformer,
    DinoVisionTransformerSSD,
    depth_anything_v2_model_args,
)
from quanta_neural_networks.depth_anything_v2.dpt import DPTHead
from quanta_neural_networks.depth_anything_v2.transform import (
    Resize,
    NormalizeImage,
    PrepareForNet,
)
from quanta_neural_networks.utils.train_utils import freeze_module


class DepthAnythingV2(nn.Module):
    """
    DepthAnything-v2 model with DINO transformer backbone.

    :param encoder: Which backbone variant to use ('vits', 'vitb', 'vitl')
    :param features: Feature output dimension for DPTHead
    :param out_channels: Output channels tuple (DPTHead configuration)
    :param use_bn: Use batch normalization
    :param image_size: Expected input image spatial size

    .. note::
        Uses DinoVisionTransformer (Meta, Apache-2.0) as backbone, designed as
        in the DepthAnything-v2 project (https://github.com/isl-org/Depth-Anything).
    """
    def __init__(
        self,
        encoder: str = "vits",
        features: int = 64,
        out_channels: tuple[int, int, int, int] = (48, 96, 192, 384),
        use_bn: bool = False,
        image_size: int = 518,
    ):
        super(DepthAnythingV2, self).__init__()

        self.intermediate_layer_idx = {
            "vits": [2, 5, 8, 11],
            "vitb": [2, 5, 8, 11],
            "vitl": [4, 11, 17, 23],
        }

        self.encoder = encoder
        self.pretrained = DinoVisionTransformer(**depth_anything_v2_model_args[encoder])

        self.depth_head = DPTHead(
            self.pretrained.embed_dim,
            features,
            use_bn,
            out_channels=out_channels,
        )

        self.resize = Resize(
            width=image_size,
            height=image_size,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        )
        self.image_size = image_size

    def forward(
        self,
        x: Float[Tensor, "batch channels height width"],
        get_feature_representation: bool = False
    ) -> Float[Tensor, "batch height width"]:
        """
        Forward pass for depth prediction from a single image input.

        :param x: Input image tensor, Float[Tensor, "batch channels height width"]
        :param get_feature_representation: Whether to also return features
        :return: Depth map, Float[Tensor, "batch height width"]
        """
        h, w = x.shape[-2:]
        x = self.resize.forward_tensor(x)

        features = self.pretrained.get_intermediate_layers(
            x,
            self.intermediate_layer_idx[self.encoder],
            reshape=True,
        )

        depth = self.depth_head(
            features, get_feature_representation=get_feature_representation
        )
        if get_feature_representation:
            depth, features = depth

        depth = F.relu(depth)

        depth = F.interpolate(
            depth, (h, w), mode="bilinear", align_corners=True
        ).squeeze(1)

        if get_feature_representation:
            return depth, features

        return depth

    @torch.no_grad()
    def infer_image(self, raw_image: np.ndarray) -> np.ndarray:
        image, (h, w) = self.image2tensor(raw_image)

        depth = self.forward(image)

        depth = F.interpolate(
            depth[:, None], (h, w), mode="bilinear", align_corners=True
        )[0, 0]

        return depth.cpu().numpy()

    def image2tensor(self, raw_image: np.ndarray) -> tuple[Float[Tensor, "batch channels height width"], tuple[int, int]]:
        transform = Compose(
            [
                self.resize,
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

        h, w = raw_image.shape[:2]

        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

        image = transform({"image": image})["image"]
        image = torch.from_numpy(image).unsqueeze(0)

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        image = image.to(device)

        return image, (h, w)


class DepthAnythingV2SSM(DepthAnythingV2):
    """
    DepthAnything variant with sequential state-space modeling (SSM) in backbone.

    :param encoder: Which backbone variant to use ('vits', 'vitb', 'vitl')
    :param state_dim: State-space dimension for SSM block
    :param features: Feature output dimension
    :param subsampling: Frame/time subsampling factor
    :param out_channels: Output channels for DPTHead
    :param use_bn: Use batch normalization
    :param image_size: Input size
    :param integrator_kwargs: Additional integrator arguments

    .. note::
        This structure combines DINO-SSD (Meta, Apache-2.0) and DepthAnything-v2 SSM.
    """
    def __init__(
        self,
        encoder: str = "vits",
        state_dim: int = 8,
        features: int = 64,
        subsampling: int | tuple[int] = 64,
        out_channels: tuple[int, int, int, int] = (48, 96, 192, 384),
        use_bn: bool = False,
        image_size: int = 518,
        **integrator_kwargs,
    ):
        super().__init__(
            encoder=encoder,
            features=features,
            out_channels=out_channels,
            use_bn=use_bn,
            image_size=image_size,
        )
        self.pretrained = DinoVisionTransformerSSD(
            state_dim=state_dim,
            subsampling=subsampling,
            **integrator_kwargs,
            **depth_anything_v2_model_args[encoder],
        )

        freeze_module(self.depth_head)

    def forward(
        self,
        photon_cube: Bool[Tensor, "h w t"],
        bocpd_gamma: float = 1e-4,
        min_window: int = 32,
        online: bool = False,
        quantile: float = 1.0,
        hot_pixel_mask: Bool[np.ndarray, "h w"] = None,
        use_tqdm: bool = False,
        clear_states: bool = True,
        **integrator_kwargs,
    ) -> tuple[Float[Tensor, "height width seq_len"], list[int]]:
        """
        Predicts depth sequence from a photon_cube input (video, time).

        :param photon_cube: Input tensor, Bool[Tensor, "height width time"]
        :param bocpd_gamma: Optional argument for integrator
        :return: Tuple (depth_tensor, time_indices)
        """
        h, w, t = photon_cube.shape

        features_ll, t_index_ll = self.pretrained.get_intermediate_layers(
            photon_cube,
            bocpd_gamma=bocpd_gamma,
            min_window=min_window,
            n=self.intermediate_layer_idx[self.encoder],
            reshape=True,
            online=online,
            quantile=quantile,
            hot_pixel_mask=hot_pixel_mask,
            use_tqdm=use_tqdm,
            clear_states=clear_states,
            **integrator_kwargs,
        )

        if online:
            depth_ll = []
            for seq_idx in range(len(t_index_ll)):
                batched_features = (
                    feature[seq_idx].unsqueeze(0) for feature in features_ll
                )
                depth = self.depth_head(batched_features).squeeze()
                depth = F.relu(depth)
                depth_ll.append(depth)
            depth_ll = torch.stack(depth_ll, dim=-1)
            depth_ll = rearrange(depth_ll, "h w seq_len -> seq_len 1 h w")
        else:
            depth_ll = self.depth_head(features_ll)
            depth_ll = F.relu(depth_ll)

        depth_ll = F.interpolate(depth_ll, (h, w), mode="bilinear", align_corners=True)
        depth_ll = rearrange(depth_ll, "seq_len 1 h w -> h w seq_len")

        return depth_ll, t_index_ll


if __name__ == "__main__":
    from loguru import logger

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = DepthAnythingV2SSM(subsampling=128)
    ckpt = torch.load(
        f"/nobackup2/vsundar4/learned_projections_ckpt/depth_anything_v2_{model.encoder}.pth",
        map_location="cpu",
    )
    model.load_state_dict(
        ckpt,
        strict=False,
    )

    model = model.to(device).eval()

    logger.info("Printing trainable params")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"{name} of shape {param.shape}")
    out_ll, t_index_ll = model(
        torch.rand(518, 518, 2048, device=device), initial_timescale=1
    )
    out_ll.sum().backward()
    logger.info(f"Out of shape {out_ll.shape}")
