import torch
from einops import rearrange
from jaxtyping import Float
from torch import nn, Tensor


class DepthLoss(nn.Module):
    def __init__(
        self,
        valid_mask: bool = True,
        loss_weight: float = 1.0,
        min_depth: float = None,
        max_depth: float = None,
        ignore_quantile: float = None,
        on_depth: bool = True,
    ):
        super().__init__()
        self.valid_mask = valid_mask
        self.loss_weight = loss_weight

        # Whether we operate on depth or disparity
        self.on_depth = on_depth

        self.max_depth = max_depth
        self.min_depth = min_depth

        if ignore_quantile:
            assert 0 <= ignore_quantile <= 1
        self.ignore_quantile = ignore_quantile

        self.eps = 1e-24  # avoid grad explode

    @property
    def min_disparity(self):
        return 1 / self.max_depth if self.max_depth else None

    @property
    def max_disparity(self):
        return 1 / self.min_depth if self.min_depth else None

    @property
    def is_disparity(self):
        return not self.on_depth

    def _get_valid_mask(self, tensor: Tensor) -> Tensor:
        """
        :param tensor: tensor to operate on
        :return:
        """
        valid_mask = None

        if self.valid_mask:
            if self.on_depth:
                valid_mask = (tensor > self.min_depth) & (tensor < self.max_depth)
            else:
                valid_mask = (tensor > self.min_disparity) & (
                    tensor < self.max_disparity
                )

        return valid_mask

    def robust_align_tensor(self, tensor: Tensor):
        """
        Align using median for shift and MAE (of median shifted) for scale.

        :param tensor: Tensor to align. H, W or H, W, T typically.
        :return:
        """
        shift = torch.median(tensor)
        aligned_disparity = tensor - shift
        scale = aligned_disparity.abs().mean()
        aligned_disparity = aligned_disparity / scale.clamp(min=self.eps)
        return aligned_disparity

    def forward(self, pred, target):
        raise NotImplementedError


class SiLogLoss(DepthLoss):
    """Scale-invariant log loss.

        This follows `AdaBins <https://arxiv.org/abs/2011.14141>`_.

    Args:
        valid_mask (bool): Whether filter invalid gt (gt > 0). Default: True.
        loss_weight (float): Weight of the loss. Default: 1.0.
        max_depth (int): When filtering invalid gt, set a max threshold. Default: None.
    """

    def __init__(
        self,
        valid_mask: bool = True,
        loss_weight: float = 1.0,
        min_depth: float = None,
        max_depth: float = None,
        lambd: float = 0.5,
    ):
        super().__init__(
            valid_mask=valid_mask,
            loss_weight=loss_weight,
            on_depth=True,
            min_depth=min_depth,
            max_depth=max_depth,
        )

        self.lambd = lambd

    def forward(self, input_tensor, target_tensor):
        valid_mask = self._get_valid_mask(target_tensor)
        if valid_mask is not None:
            input_tensor = input_tensor[valid_mask]
            target_tensor = target_tensor[valid_mask]

        # See: https://arxiv.org/pdf/2311.03938
        diff_log = torch.log(input_tensor + self.eps) - torch.log(
            target_tensor + self.eps
        )

        loss = torch.sqrt(
            torch.pow(diff_log, 2).mean() - self.lambd * torch.pow(diff_log.mean(), 2)
        )
        return self.loss_weight * loss


class MAELoss(DepthLoss):
    """AffineInvariantLoss.

    Source: https://arxiv.org/pdf/2401.10891v1

    Args:
        valid_mask (bool): Whether filter invalid gt (gt > 0). Default: True.
        loss_weight (float): Weight of the loss. Default: 1.0.
        max_depth (int): When filtering invalid gt, set a max threshold. Default: None.
        min_depth (int): When filtering invalid gt, set a min threshold. Default: None.
    """

    def __init__(
        self,
        valid_mask: bool = True,
        loss_weight: float = 1.0,
        min_depth: float = None,
        max_depth: float = None,
        affine_invariant: bool = True,
        ignore_quantile: float = None,
        on_depth: bool = False,
    ):
        super().__init__(
            valid_mask=valid_mask,
            loss_weight=loss_weight,
            min_depth=min_depth,
            max_depth=max_depth,
            ignore_quantile=ignore_quantile,
            on_depth=on_depth,
        )
        self.affine_invariant = affine_invariant
        self.eps = 1e-6  # avoid grad exploding

    def forward(
        self,
        pred_tensor: Float[Tensor, "h w t"],
        target_tensor: Float[Tensor, "h w t"],
    ):
        """
        Assumes depth tensors
        :param pred_tensor:
        :param target_tensor:
        :return:
        """
        valid_mask = self._get_valid_mask(target_tensor)

        if self.affine_invariant:
            pred_tensor = self.robust_align_tensor(pred_tensor)
            target_tensor = self.robust_align_tensor(target_tensor)

        deviation = (pred_tensor - target_tensor).abs()

        if valid_mask is not None:
            deviation = deviation[valid_mask]

        if self.ignore_quantile:
            # Ignore top contributing regions
            threshold = torch.quantile(deviation, self.ignore_quantile)
            deviation = deviation[deviation < threshold]

        return self.loss_weight * deviation.mean()


class GradientLoss(DepthLoss):
    """Gradient loss. First order.
    Matches gradients of inputs to targets.
    Can be applied to depth or disparities.

    Source: https://arxiv.org/pdf/2401.10891v1

    Args:
        valid_mask (bool): Whether filter invalid gt (gt > 0). Default: True.
        loss_weight (float): Weight of the loss. Default: 1.0.
        max_depth (int): When filtering invalid gt, set a max threshold. Default: None.
    """

    def __init__(
        self,
        valid_mask: bool = True,
        loss_weight: float = 1.0,
        min_depth: float = None,
        max_depth: float = None,
        affine_invariant: bool = False,
        scale_invariant: bool = False,
        ignore_quantile: float = None,
        on_depth: bool = False,
        num_scales: int = 3,
    ):
        super().__init__(
            valid_mask=valid_mask,
            loss_weight=loss_weight,
            min_depth=min_depth,
            max_depth=max_depth,
            ignore_quantile=ignore_quantile,
            on_depth=on_depth,
        )
        self.num_scales = num_scales
        self.affine_invariant = affine_invariant
        self.scale_invariant = scale_invariant

    def get_tensor_scales(self, tensor: Float[Tensor, "h w t"], num_scales: int):
        tensor_ll = [tensor]
        for scale_idx in range(1, num_scales):
            if tensor.ndim == 2:
                tensor_scaled = rearrange(tensor, "h w -> 1 1 h w")
            elif tensor.ndim == 3:
                tensor_scaled = rearrange(tensor, "h w t -> t 1 h w")
            else:
                raise NotImplementedError

            # tensor_scaled = F.interpolate(
            #     tensor_scaled,
            #     scale_factor=1 / pow(2, scale_idx),
            # )
            tensor_scaled = tensor_scaled[
                :, :, :: pow(2, scale_idx), :: pow(2, scale_idx)
            ]

            if tensor.ndim == 2:
                tensor_scaled = rearrange(tensor_scaled, "1 1 h w -> h w")
            elif tensor.ndim == 3:
                tensor_scaled = rearrange(tensor_scaled, "t 1 h w -> h w t")

            tensor_ll.append(tensor_scaled)
        return tensor_ll

    def forward(
        self,
        input_tensor: Float[Tensor, "h w t"],
        target_tensor: Float[Tensor, "h w t"],
    ):
        """
        Assumes depth tensors
        :param input_tensor:
        :param target_tensor:
        :return:
        """
        input_downscaled_ll = self.get_tensor_scales(
            input_tensor, num_scales=self.num_scales
        )
        target_downscaled_ll = self.get_tensor_scales(
            target_tensor, num_scales=self.num_scales
        )

        gradient_loss = 0
        for input_downscaled, target_downscaled in zip(
            input_downscaled_ll, target_downscaled_ll
        ):
            mask = self._get_valid_mask(target_downscaled)

            num = mask.sum() if mask is not None else target_downscaled.numel()

            if self.affine_invariant:
                input_downscaled = self.robust_align_tensor(input_downscaled)
                target_downscaled = self.robust_align_tensor(target_downscaled)
            elif self.scale_invariant:
                input_downscaled = torch.log(input_downscaled + self.eps)
                target_downscaled = torch.log(target_downscaled + self.eps)

            diff = input_downscaled - target_downscaled

            if mask is not None:
                diff = diff * mask

            v_gradient = torch.abs(diff[:-2, :] - diff[2:, :])
            if mask is not None:
                v_mask = torch.mul(mask[:-2, :], mask[2:, :])
                v_gradient = torch.mul(v_gradient, v_mask)

            h_gradient = torch.abs(diff[:, :-2] - diff[:, 2:])
            if mask is not None:
                h_mask = torch.mul(mask[:, :-2], mask[:, 2:])
                h_gradient = torch.mul(h_gradient, h_mask)

            gradient_loss += (h_gradient.sum() + v_gradient.sum()) / num

        return self.loss_weight * gradient_loss / self.num_scales
