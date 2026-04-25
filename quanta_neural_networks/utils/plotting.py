"""
Plotting and visualization helpers for analysis and result reporting.
"""
from dataclasses import dataclass

import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.pyplot import colormaps

SAVEFIG_PDF_KWARGS = dict(dpi=150, transparent=True, bbox_inches="tight")


def mpl_color_cycler(i: int):
    """
    Access the i-th default matplotlib color for plotted lines/markers.
    :param i: Color index
    :return: Hex or RGB color as string/int
    """
    return plt.rcParams["axes.prop_cycle"].by_key()["color"][i]


def arrowed_spines(ax, use_dark: bool = False):
    """
    Add centered axes with arrowheads to a matplotlib axes.
    :param ax: The axes to modify
    :param use_dark: Use white instead of black for markers
    """

    # Move the left and bottom spines to x = 0 and y = 0, respectively.
    ax.spines[["left", "bottom"]].set_position(("data", 0))
    # Hide the top and right spines.
    ax.spines[["top", "right"]].set_visible(False)

    ax.tick_params(top=False, right=False)

    # Draw arrows (as black triangles: ">k"/"^k") at the end of the axes.  In each
    # case, one of the coordinates (0) is a data coordinate (i.e., y = 0 or x = 0,
    # respectively) and the other one (1) is an axes coordinate (i.e., at the very
    # right/top of the axes).  Also, disable clipping (clip_on=False) as the marker
    # actually spills out of the axes.
    color = "k" if not use_dark else "w"
    ax.plot(1, 0, f">{color}", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, f"^{color}", transform=ax.get_xaxis_transform(), clip_on=False)


def color_depth(normalized_tensor: torch.Tensor, colors: colormaps) -> torch.Tensor:
    """
    Map normalized tensor values to color images using a colormap.
    :param normalized_tensor: Input normalized values [0,1]
    :param colors: Colormap to use
    :return: Colored tensor
    """
    return colors(normalized_tensor)[..., :3]


params = {
    # "legend.fontsize": 10,
    "figure.figsize": (5.75, 2.5),  # (3.75, 4.25),  # (5.75, 3.025),
    "font.size": 11,
    # "axes.labelsize": 18,
    # "axes.titlesize": 16,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
}


@dataclass
class PlotInfo:
    """
    Container for matplotlib plot/axes handles and settings.
    """
    ylabel: str
    fig: Figure
    ax: Axes
    ylim: tuple[int, int] = (None, None)
