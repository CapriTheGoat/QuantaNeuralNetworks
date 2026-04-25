"""
Loss functions for point tracking.

This module contains various loss functions used for training point tracking models,
including balanced cross-entropy loss, sequence loss, and score map loss.
"""

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor
from torch.nn import functional as F

from quanta_neural_networks.ops.array_ops import reduce_masked_mean


def balanced_ce_loss(pred: Float[Tensor, "..."], gt: Float[Tensor, "..."], valid: Float[Tensor, "..."] | None = None) -> Float[Tensor, ""]:
    """
    Compute balanced cross-entropy loss for binary classification.
    
    :param pred: Predicted logits
    :param gt: Ground truth binary labels
    :param valid: Validity mask (optional)
    :return: Balanced cross-entropy loss
    """
    assert pred.shape == gt.shape
    if valid is not None:
        assert valid.shape == pred.shape
    else:
        valid = torch.ones_like(gt)

    pos = (gt > 0.95).float()
    neg = (gt < 0.05).float()
    label = pos * 2.0 - 1.0
    a = -label * pred
    b = F.relu(a)
    loss = b + torch.log(torch.exp(-b) + torch.exp(a - b))

    pos_loss = reduce_masked_mean(loss, pos * valid)
    neg_loss = reduce_masked_mean(loss, neg * valid)

    balanced_loss = pos_loss + neg_loss

    return balanced_loss


def sequence_loss(
    flow_gt: Float[Tensor, "num_frame num_points coords"],
    flow_preds: Float[Tensor, "num_preds num_frame num_points coords"],
    valids_gt: Float[Tensor, "num_frame num_points"],
    valids_pred: Float[Tensor, "num_frame num_points"] | None = None,
    gamma: float = 0.8,
    bce_weight: float = 0.5,
) -> Float[Tensor, ""]:
    """
    Compute sequence loss over multiple flow predictions.
    
    :param flow_gt: Ground truth flow of shape (num_frame, num_points, coords)
    :param flow_preds: Predicted flows of shape (num_preds, num_frame, num_points, coords)
    :param valids_gt: Ground truth validity mask of shape (num_frame, num_points)
    :param valids_pred: Predicted validity mask of shape (num_frame, num_points)
    :param gamma: Exponential decay factor for multi-scale predictions
    :param bce_weight: Weight for binary cross-entropy loss
    :return: Sequence loss value
    """
    flow_loss = 0.0
    num_predictions = len(flow_preds)

    for i in range(num_predictions):
        i_weight = gamma ** (num_predictions - i - 1)
        flow_pred = flow_preds[i]
        l1_loss: Float[Tensor, "num_frame num_points"] = (flow_pred - flow_gt).norm(
            p=1, dim=-1
        )
        flow_loss += i_weight * reduce_masked_mean(l1_loss, valids_gt)

    flow_loss = flow_loss / len(flow_preds)
    if valids_pred is not None:
        flow_loss += bce_weight * F.binary_cross_entropy_with_logits(
            valids_pred, valids_gt
        )
    return flow_loss


def score_map_loss(feature_map_score_ll: Float[Tensor, "num_frame num_iter num_points h w"], gt_trajectory: Float[Tensor, "num_frame num_points coords"], valids: Float[Tensor, "num_frame num_points"]) -> Float[Tensor, ""]:
    """
    Compute score map loss for feature map predictions.
    
    :param feature_map_score_ll: Feature map scores of shape (num_frame, num_iter, num_points, h, w)
    :param gt_trajectory: Ground truth trajectory of shape (num_frame, num_points, coords)
    :param valids: Validity mask of shape (num_frame, num_points)
    :return: Score map loss value
    """
    _, _, _, fmap_h, fmap_w = feature_map_score_ll.shape
    fcp_ = rearrange(
        feature_map_score_ll,
        "num_frame num_iter num_points h w -> (num_frame num_points) num_iter h w",
    )
    xy_ = rearrange(
        gt_trajectory, "num_frame num_points coords -> (num_frame num_points) coords"
    ).long()

    valid_ = rearrange(valids, "num_frame num_points -> (num_frame num_points)")
    x_, y_ = xy_[:, 0], xy_[:, 1]  # BSN
    ind = (
        (x_ >= 0)
        & (x_ <= (fmap_w - 1))
        & (y_ >= 0)
        & (y_ <= (fmap_h - 1))
        & (valid_ > 0)
    )  # BSN
    fcp_ = fcp_[ind]  # N_,I,H8,W8
    xy_ = xy_[ind]  # N_,2
    N_ = fcp_.shape[0]

    # make gt with ones at the rounded spatial inds in here
    gt_ = torch.zeros_like(fcp_)  # N_,I,H8,W8
    for n in range(N_):
        gt_[n, :, xy_[n, 1], xy_[n, 0]] = 1
    ## ce
    fcp_ = fcp_.flatten()
    gt_ = gt_.flatten()
    ce_loss = balanced_ce_loss(fcp_, gt_)
    return ce_loss
