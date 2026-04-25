"""
Training utilities for checkpoints, freezing modules, batchnorm fusion, and simulation routines.
"""
from pathlib import Path
from typing import Dict

import torch
from einops import repeat
from loguru import logger
from natsort import natsorted
from omegaconf import ListConfig
from torch import nn as nn, Tensor


def as_tuple(x, size):
    """
    Convert input to tuple of desired size.
    :param x: List, tuple, ListConfig, or scalar to expand.
    :param size: Repeat factor for scalar/other.
    :return: Tuple of length size.
    """
    return tuple(x) if (type(x) in [list, tuple, ListConfig]) else (x,) * size


def as_tuples(*args, size):
    """
    Expand one or more inputs as tuples of size 'size'.
    :param args: Arguments to expand
    :param size: Length of output tuples
    :return: Tuple of tuples
    """
    return tuple(as_tuple(arg, size=size) for arg in args)


def resume_or_finetune(
    model: torch.nn.Module,
    optimizer: torch.optim,
    ckpt_cfg: Dict,
    scheduler: torch.optim.lr_scheduler = None,
):
    """
    Resume or finetune a model from the latest checkpoint if available, else init.

    :param model: Model to restore state on
    :param optimizer: Optimizer to restore
    :param ckpt_cfg: Folder/path/flags for checkpoint search
    :param scheduler: LR scheduler (optional)
    :return: (epoch_start, global_step)
    """
    global_step = 0
    epoch_start = 0

    # Create if path doesn't exist
    Path(ckpt_cfg.folder).mkdir(exist_ok=True, parents=True)

    ckpt_file_ll = list(Path(ckpt_cfg.folder).rglob("*.pth"))
    ckpt_file_ll = natsorted(ckpt_file_ll, reverse=True, key=lambda u: u.name)

    # Load latest ckpt (resume / finetune)
    if ckpt_cfg.resume and len(ckpt_file_ll) != 0:
        ckpt_file_latest = ckpt_file_ll[0]
        logger.info(f"Resuming from {ckpt_file_latest}")
        ckpt = torch.load(ckpt_file_latest, map_location="cpu", weights_only=False)

        # Load model, optimizer, scheduler
        model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])

        if scheduler is not None:
            scheduler.load_state_dict(ckpt["scheduler"])

        global_step = ckpt["global_step"]

    elif ckpt_cfg.get("finetune_from") and Path(ckpt_cfg.get("finetune_from")).exists():
        logger.info(f"Finetuning from {ckpt_cfg.finetune_from}")

        ckpt = torch.load(
            ckpt_cfg.finetune_from, map_location="cpu", weights_only=False
        )

        # Load model, optimizer, scheduler
        missing_keys, unexpected_keys = model.load_state_dict(
            ckpt["model"], strict=False
        )
        logger.info(f"Missing keys {missing_keys}")
        logger.info(f"Unexpected keys {unexpected_keys}")
    return epoch_start, global_step


def load_checkpoint(
    model: torch.nn.Module,
    ckpt_folder: Path | str = None,
    ckpt_file: Path = None,
    ckpt_key: str = "model",
    strict: bool = True,
    weights_only: bool = False,
):
    """
    Load state dict from checkpoint file, with optional strictness and weights-only mode.
    :param model: PyTorch model
    :param ckpt_folder: Folder to look for .pth files
    :param ckpt_file: Specific file to load
    :param ckpt_key: Dict key in checkpoint for model weights
    :param strict: Set strict loading
    :param weights_only: Only load weights (no optimizer/scheduler)
    """
    if not ckpt_file or not Path(ckpt_file).exists():
        ckpt_folder = Path(ckpt_folder)
        ckpt_file_ll = list(ckpt_folder.rglob("*.pth"))
        if len(ckpt_file_ll):
            ckpt_file = natsorted(ckpt_file_ll, reverse=True, key=lambda u: u.name)[0]
        else:
            logger.warning(f"No ckpt found at {ckpt_folder}")
            return

    logger.info(f"Loading model params from {ckpt_file}")
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=weights_only)
    if ckpt_key:
        if ckpt_key not in ckpt:
            raise KeyError(f"no key '{ckpt_key}' in ckpt")
        ckpt = ckpt.get(ckpt_key)
    ckpt = {k: v for k, v in ckpt.items() if not k.startswith("integrator.")}
    missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=strict)
    if not strict:
        logger.info(f"Missing keys {missing_keys}")
        logger.info(f"Unexpected keys {unexpected_keys}")


def fuse_bn_in_sequential(sequential_module: nn.Module):
    """
    In-place fuse Conv2d+BatchNorm2d modules in nn.Sequential blocks.
    :param sequential_module: Module or Sequential to fuse
    :return: Module with fusions (and replaced with Identity)
    """
    module_output = sequential_module
    if isinstance(sequential_module, (nn.Sequential,)):
        for idx in range(len(sequential_module) - 1):
            if not isinstance(
                sequential_module[idx],
                nn.Conv2d,
            ) or not isinstance(sequential_module[idx + 1], nn.BatchNorm2d):
                continue
            conv = sequential_module[idx]
            bn = sequential_module[idx + 1]

            invstd = 1 / torch.sqrt(bn.running_var + bn.eps)
            conv.weight.data = (
                conv.weight
                * bn.weight[:, None, None, None]
                * invstd[:, None, None, None]
            )
            if conv.bias is None:
                conv.bias = nn.Parameter(torch.zeros(conv.out_channels))
            conv.bias.data = (
                conv.bias - bn.running_mean
            ) * bn.weight * invstd + bn.bias
            sequential_module[idx + 1] = nn.Identity()

    for name, child in sequential_module.named_children():
        module_output.add_module(name, fuse_bn_in_sequential(child))
    del sequential_module
    return module_output


def freeze_module(module: nn.Module):
    """
    Set requires_grad=False for all params in module (freeze).
    """
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_module(module: nn.Module):
    """
    Set requires_grad=True for all params in module (unfreeze).
    """
    for p in module.parameters():
        p.requires_grad = True


@torch.no_grad()
def simulate_photon_cube(
    intensity_ll,
    oversampling: int = 1,
    num_time_step: int = None,
    max_probability: float = 1.0,
):
    """
    Stochastic binary sampling of a simulated photon cube, for Monte Carlo inference.
    :param intensity_ll: Float tensor data (h w t or b h w t)
    :param oversampling: #times to repeat each time-step
    :param num_time_step: Truncate/slice output at this length if not None
    :param max_probability: Scale input by this before sampling
    :return: Binary (float) simulated photon cube
    """
    if oversampling > 1:
        if intensity_ll.ndim == 4:
            photon_cube_intensity_ll = repeat(
                intensity_ll,
                "b h w t -> b h w (t num_repeat)",
                num_repeat=oversampling,
            )
        elif intensity_ll.ndim == 3:
            photon_cube_intensity_ll = repeat(
                intensity_ll,
                "h w t -> h w (t num_repeat)",
                num_repeat=oversampling,
            )
        else:
            raise AssertionError(f"Found {intensity_ll.shape}")
    else:
        photon_cube_intensity_ll = intensity_ll

    photon_cube_intensity_ll: Tensor = (
        photon_cube_intensity_ll[..., :num_time_step] * max_probability
    )
    photon_cube = (
        torch.rand(
            *photon_cube_intensity_ll.shape,
            device=photon_cube_intensity_ll.device,
        )
        < photon_cube_intensity_ll
    ).float()
    return photon_cube


