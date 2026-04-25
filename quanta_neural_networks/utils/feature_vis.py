"""
Adapted from https://github.com/mhamilton723/FeatUp/blob/6b5a6c0e91f75e69194807128dcbc39c3084a30d/featup/util.py#L174
"""

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor


class TorchPCA:
    def __init__(self, n_components: int):
        self.n_components = n_components

    def fit(self, X: Float[Tensor, "batch dim"]):
        # self.mean_ = torch.zeros_like(X.mean(dim=0))
        self.mean_ = X.mean(dim=0)
        unbiased = X - self.mean_.unsqueeze(0)
        U, S, V = torch.pca_lowrank(
            unbiased, q=self.n_components, center=False, niter=4
        )
        self.components_ = V.T
        self.singular_values_ = S
        return self

    def transform(self, X: Float[Tensor, "batch dim"]):
        t0 = X - self.mean_.unsqueeze(0)
        projected = t0 @ self.components_.T
        return projected


def cast_features_to_rgb(
    image_feature: Float[Tensor, "batch channels height width"],
    dim: int = 3,
    fit_pca: TorchPCA = None,
    max_samples: int = None,
) -> tuple[Float[Tensor, "batch 3 height width"], TorchPCA]:
    batch, channels, height, width = image_feature.shape

    flattened_feature = rearrange(
        image_feature, "batch channels height width -> (batch height width) channels"
    )

    # Subsample the data if max_samples is set and the number of samples exceeds max_samples
    if max_samples is not None and flattened_feature.shape[0] > max_samples:
        indices = torch.randperm(flattened_feature.shape[0])[:max_samples]
        flattened_feature = flattened_feature[indices]

    if fit_pca is None:
        fit_pca = TorchPCA(n_components=dim).fit(flattened_feature)

    flattened_feature_reduced = fit_pca.transform(flattened_feature)
    flattened_feature_reduced -= flattened_feature_reduced.min(
        dim=0, keepdim=True
    ).values
    flattened_feature_reduced /= flattened_feature_reduced.max(
        dim=0, keepdim=True
    ).values

    feature_reduced = rearrange(
        flattened_feature_reduced,
        "(batch height width) channel -> " "batch channel height width",
        batch=batch,
        height=height,
        width=width,
    )

    return feature_reduced, fit_pca
