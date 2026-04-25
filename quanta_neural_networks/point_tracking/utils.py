"""
Utility functions for point tracking visualization and processing.

This module contains helper functions for point tracking including visualization,
coordinate estimation, and various image processing utilities.
"""

from math import floor
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from einops import reduce, rearrange, repeat
from jaxtyping import Float, Bool
from loguru import logger
from matplotlib import cm
from torch import Tensor
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter


def spatial_softmax(tensor: Float[Tensor, "batch channels height width"]) -> Float[Tensor, "batch channels height width"]:
    """
    Apply spatial softmax to a tensor.
    
    :param tensor: Input tensor of shape (batch, channels, height, width)
    :return: Spatial softmax applied tensor of same shape
    """
    batch, channels, height, width = tensor.shape
    tensor_flattened = rearrange(tensor, "b c h w -> b c (h w)")
    tensor_flattened = F.softmax(tensor_flattened, dim=-1)  # shape (B, 1, H*W)
    output = rearrange(tensor_flattened, "b c (h w) -> b c h w", h=height, w=width)
    return output


def get_initial_coords_estimate(
    cost_volume: Float[Tensor, "num_frame num_points h w"], argmax_radius: float
) -> Float[Tensor, "num_frame num_points coords"]:
    """
    Get initial coordinate estimates from cost volume using spatial softmax.
    
    :param cost_volume: Cost volume tensor of shape (num_frame, num_points, h, w)
    :param argmax_radius: Radius for filtering around argmax coordinates
    :return: Estimated coordinates of shape (num_frame, num_points, coords)
    """
    num_frame, num_points, feature_h, feature_w = cost_volume.shape
    device = cost_volume.device
    cost_volume = spatial_softmax(cost_volume)

    # Find argmax coords
    argmax_flat = torch.argmax(
        rearrange(
            cost_volume, "num_frame num_points h w -> num_frame num_points (h w)"
        ),
        dim=-1,
    )

    # Average spatially, don't take argmax
    grid_y, grid_x = torch.meshgrid(
        [
            torch.arange(feature_h, device=device),
            torch.arange(feature_w, device=device),
        ],
        indexing="ij",
    )
    grid: Float[Tensor, "h w num_coords"] = torch.stack(
        (grid_x + 0.5, grid_y + 0.5), -1
    )

    argmax_coords = rearrange(grid, "h w num_coords -> (h w) num_coords")[argmax_flat]
    argmax_coords = rearrange(
        argmax_coords,
        "num_frame num_points num_coords -> num_frame num_points 1 1 num_coords",
    )

    # Filter cost volume around the argmax coords
    grid = rearrange(grid, "h w num_coords -> 1 1 h w num_coords")

    cost_volume_mask = (
        torch.norm((grid - argmax_coords).float(), dim=-1) <= argmax_radius
    )

    cost_volume = cost_volume_mask * cost_volume
    cost_volume_sum = reduce(
        cost_volume, "num_frame num_points h w -> num_frame num_points", "sum"
    )

    coords = reduce(
        cost_volume.unsqueeze(-1) * grid,
        "num_frame num_points h w num_coords -> num_frame num_points num_coords",
        "sum",
    )
    coords /= cost_volume_sum.clamp(min=1e-8).unsqueeze(-1)

    return coords


def draw_circle(rgb: Image.Image, coord: tuple[int, int], radius: int, color: tuple[int, int, int] = (255, 0, 0), visible: bool = True, color_alpha: int | None = None) -> Image.Image:
    """
    Draw a circle on an RGB image.
    
    :param rgb: PIL Image to draw on
    :param coord: Center coordinates (x, y)
    :param radius: Circle radius
    :param color: RGB color tuple
    :param visible: Whether to fill the circle
    :param color_alpha: Alpha value for transparency
    :return: Modified PIL Image
    """
    # Create a draw object
    draw = ImageDraw.Draw(rgb)
    # Calculate the bounding box of the circle
    left_up_point = (coord[0] - radius, coord[1] - radius)
    right_down_point = (coord[0] + radius, coord[1] + radius)
    # Draw the circle
    color = tuple(list(color) + [color_alpha if color_alpha is not None else 255])
    draw.ellipse(
        [left_up_point, right_down_point],
        fill=tuple(color) if visible else None,
        outline=tuple(color),
    )
    return rgb


def draw_line(rgb: Image.Image, coord_y: tuple[int, int], coord_x: tuple[int, int], color: tuple[int, int, int], linewidth: int) -> Image.Image:
    """
    Draw a line on an RGB image.
    
    :param rgb: PIL Image to draw on
    :param coord_y: Start coordinates (x, y)
    :param coord_x: End coordinates (x, y)
    :param color: RGB color tuple
    :param linewidth: Line width in pixels
    :return: Modified PIL Image
    """
    draw = ImageDraw.Draw(rgb)
    draw.line(
        (coord_y[0], coord_y[1], coord_x[0], coord_x[1]),
        fill=tuple(color),
        width=linewidth,
    )
    return rgb


def add_weighted(rgb: np.ndarray, alpha: float, original: np.ndarray, beta: float, gamma: float) -> np.ndarray:
    """
    Add weighted combination of two images.
    
    :param rgb: First image array
    :param alpha: Weight for first image
    :param original: Second image array
    :param beta: Weight for second image
    :param gamma: Bias term
    :return: Weighted combination as uint8 array
    """
    return (
        rgb.astype(np.float32) * alpha + original.astype(np.float32) * beta + gamma
    ).astype("uint8")


class Visualizer:
    """
    Visualizer for point tracking results.
    
    This class provides functionality to visualize point tracking results
    by drawing trajectories on intensity sequences and saving videos.
    """
    
    def __init__(
        self,
        save_dir: str | Path | None = None,
        fps: int = 10,
        cmap_name: str = "rainbow",  # 'cool', 'optical_flow'
        linewidth: int = 1,
        show_first_frame: int = 0,
        tracks_leave_trace: int = -1,  # -1 for infinite
    ) -> None:
        """
        Initialize the visualizer.
        
        :param save_dir: Directory to save visualization videos
        :param fps: Frames per second for output videos
        :param cmap_name: Color map name for trajectory visualization
        :param linewidth: Line width for drawing trajectories
        :param show_first_frame: Number of times to repeat first frame
        :param tracks_leave_trace: Number of frames to show in trajectory trace (-1 for infinite)
        """
        self.cmap_name = cmap_name

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True, parents=True)
        self.save_dir = save_dir

        if cmap_name == "rainbow":
            self.color_map = cm.get_cmap("gist_rainbow")
        elif cmap_name == "cool":
            self.color_map = cm.get_cmap(cmap_name)
        self.show_first_frame = show_first_frame
        self.grayscale = True
        self.tracks_leave_trace = tracks_leave_trace
        self.linewidth = linewidth
        self.fps = fps

    def visualize(
        self,
        photon_cube: Float[Tensor, "h w t"],  # (B,T,C,H,W)
        pred_trajectory: Float[Tensor, "num_frame num_points coords"],  # (B,T,N,2)
        visibility: Bool[Tensor, "num_frame num_points"] | None = None,  # (B, T, N, 1) bool
        gt_trajectory: Float[Tensor, "num_frame num_points coords"] | None = None,  # (B,T,N,2)
        filename: str = "video",
        writer: SummaryWriter | None = None,
        step: int = 0,
        tag: str = "",
        query_frame: int = 0,
        opacity: float = 1.0,
    ) -> Float[Tensor, "height width channels num_frame"]:
        """
        Visualize point tracking results on intensity sequence.
        
        :param photon_cube: Intensity sequence of shape (h, w, t)
        :param pred_trajectory: Predicted trajectories of shape (num_frame, num_points, coords)
        :param visibility: Visibility mask of shape (num_frame, num_points)
        :param gt_trajectory: Ground truth trajectories of shape (num_frame, num_points, coords)
        :param filename: Base filename for saving video
        :param writer: TensorBoard writer for logging
        :param step: Step number for logging
        :param tag: Tag for TensorBoard logging
        :param query_frame: Query frame index
        :param opacity: Opacity for trajectory visualization
        :return: Visualized video sequence
        """
        num_frame = pred_trajectory.shape[0]
        h, w, t = photon_cube.shape
        img_ll = reduce(
            photon_cube.float()[..., : floor(t / num_frame) * num_frame],
            "h w (num_frame t) -> h w num_frame",
            "mean",
            num_frame=num_frame,
        )

        color_alpha = int(opacity * 255)
        pred_trajectory = pred_trajectory

        img_ll = self.draw_tracks_on_photon_cube(
            img_ll=img_ll,
            pred_trajectory=pred_trajectory,
            visibility=visibility,
            gt_trajectory=gt_trajectory,
            query_frame=query_frame,
            color_alpha=color_alpha,
        )
        self.save_video(img_ll, filename=filename, writer=writer, step=step, tag=tag)
        return img_ll

    def save_video(self, img_ll, filename, writer=None, step=0, tag: str = ""):
        video = rearrange(img_ll, "h w c num_frame -> 1 num_frame c h w ")

        if writer:
            writer.add_images(
                tag,
                video.squeeze(0),
                global_step=step,
                dataformats="NCHW",
            )
        if self.save_dir:
            wide_list = list(video.unbind(1))
            wide_list = [wide[0].permute(1, 2, 0).cpu().numpy() for wide in wide_list]

            # Prepare the video file path
            save_path = self.save_dir / f"{filename}.mp4"

            # Create a writer object
            with imageio.get_writer(save_path, fps=self.fps) as video_writer:
                # Write frames to the video file
                for frame in wide_list[2:-1]:
                    video_writer.append_data(frame)

            logger.info(f"Video saved to {save_path}")

    def draw_tracks_on_photon_cube(
        self,
        img_ll: Float[Tensor, "h w num_frame"],  # (B,T,C,H,W)
        pred_trajectory: Float[Tensor, "num_frame num_points coords"],  # (B,T,N,2)
        visibility: Bool[Tensor, "num_frame num_points"] = None,  # (B, T, N, 1) bool
        gt_trajectory: Float[Tensor, "num_frame num_points coords"] = None,  # (B,T,N,2)
        query_frame=0,
        color_alpha: int = 255,
    ):
        num_frame, num_points, num_coords = pred_trajectory.shape
        assert num_coords == 2
        pred_trajectory = pred_trajectory.long().numpy(force=True)
        if gt_trajectory is not None:
            gt_trajectory = gt_trajectory.long().numpy(force=True)

        res_video = (img_ll.clone().numpy(force=True) * 255).astype(np.uint8)
        res_video = repeat(res_video, "h w t -> h w c t", c=3)
        vector_colors = np.zeros((num_frame, num_points, 3))

        if self.cmap_name == "optical_flow":
            import flow_vis

            vector_colors = flow_vis.flow_to_color(
                pred_trajectory - pred_trajectory[query_frame][None]
            )
        else:
            if self.cmap_name == "rainbow":
                y_min, y_max = (
                    pred_trajectory[query_frame, :, 1].min(),
                    pred_trajectory[query_frame, :, 1].max(),
                )
                norm = plt.Normalize(y_min, y_max)
                for point_idx in range(num_points):
                    if isinstance(query_frame, torch.Tensor):
                        query_frame_ = query_frame[point_idx]
                    else:
                        query_frame_ = query_frame
                    color = self.color_map(
                        norm(pred_trajectory[query_frame_, point_idx, 1])
                    )
                    color = np.array(color[:3])[None] * 255
                    vector_colors[:, point_idx] = np.repeat(color, num_frame, axis=0)
            else:
                # color changes with time
                for frame_idx in range(num_frame):
                    color = (
                        np.array(self.color_map(frame_idx / num_frame)[:3])[None] * 255
                    )
                    vector_colors[frame_idx] = np.repeat(color, num_points, axis=0)

        #  draw tracks
        if self.tracks_leave_trace != 0:
            for frame_idx in range(query_frame + 1, num_frame):
                first_ind = (
                    max(0, frame_idx - self.tracks_leave_trace)
                    if self.tracks_leave_trace >= 0
                    else 0
                )
                curr_tracks = pred_trajectory[first_ind : frame_idx + 1]
                curr_colors = vector_colors[first_ind : frame_idx + 1]

                if gt_trajectory is not None:
                    res_video[..., frame_idx] = self._draw_gt_tracks(
                        res_video[..., frame_idx],
                        gt_trajectory[first_ind : frame_idx + 1],
                    )

                res_video[..., frame_idx] = self._draw_pred_trajectory(
                    res_video[..., frame_idx],
                    curr_tracks,
                    curr_colors,
                )

        #  draw points
        for frame_idx in range(num_frame):
            img = Image.fromarray(res_video[..., frame_idx])
            for point_idx in range(num_points):
                coord = (
                    pred_trajectory[frame_idx, point_idx, 0],
                    pred_trajectory[frame_idx, point_idx, 1],
                )
                visibile = True
                if visibility is not None:
                    visibile = visibility[frame_idx, point_idx].item()
                if coord[0] != 0 and coord[1] != 0:
                    img = draw_circle(
                        img,
                        coord=coord,
                        radius=int(self.linewidth * 2),
                        color=vector_colors[frame_idx, point_idx].astype(int),
                        visible=visibile,
                        color_alpha=color_alpha,
                    )
            res_video[..., frame_idx] = np.array(img)

        #  construct the final rgb sequence
        if self.show_first_frame > 0:
            torch.cat(
                (
                    repeat(
                        res_video[..., 0],
                        "h w c -> h w c num_repeat",
                        num_repeat=self.show_first_frame,
                    ),
                    res_video[..., 1:],
                )
            )

        return torch.from_numpy(res_video)

    def _draw_pred_trajectory(
        self,
        img: Float[np.ndarray, "h w c"],
        pred_trajectory: Float[np.ndarray, "num_frame num_points coords"],  # T x 2
        vector_colors: np.ndarray,
    ):
        num_frame, num_points, _ = pred_trajectory.shape
        img = Image.fromarray(img)

        for frame_idx in range(num_frame - 1):
            vector_color = vector_colors[frame_idx]
            original = img.copy()
            alpha = (frame_idx / num_frame) ** 2
            for point_idx in range(num_points):
                coord_y = (
                    int(pred_trajectory[frame_idx, point_idx, 0]),
                    int(pred_trajectory[frame_idx, point_idx, 1]),
                )
                coord_x = (
                    int(pred_trajectory[frame_idx + 1, point_idx, 0]),
                    int(pred_trajectory[frame_idx + 1, point_idx, 1]),
                )
                if coord_y[0] != 0 and coord_y[1] != 0:
                    img = draw_line(
                        img,
                        coord_y,
                        coord_x,
                        vector_color[point_idx].astype(int),
                        self.linewidth,
                    )
            if self.tracks_leave_trace > 0:
                img = Image.fromarray(
                    add_weighted(np.array(img), alpha, np.array(original), 1 - alpha, 0)
                )
        img = np.array(img)
        return img

    def _draw_gt_tracks(
        self,
        img: Float[np.ndarray, "h w c"],
        gt_tracks: Float[np.ndarray, "num_frame num_points coords"],  # T x 2
    ):
        num_frame, num_points, _ = gt_tracks.shape
        color = np.array((0, 0, 211))
        img = Image.fromarray(img)
        for frame_idx in range(num_frame):
            for point_idx in range(num_points):
                gt_point = gt_tracks[frame_idx, point_idx]
                #  draw a blue cross
                if gt_point[0] > 0 and gt_point[1] > 0:
                    length = self.linewidth * 3
                    coord_y = (int(gt_point[0]) + length, int(gt_point[1]) + length)
                    coord_x = (int(gt_point[0]) - length, int(gt_point[1]) - length)
                    img = draw_line(
                        img,
                        coord_y,
                        coord_x,
                        color,
                        self.linewidth,
                    )
                    coord_y = (int(gt_point[0]) - length, int(gt_point[1]) + length)
                    coord_x = (int(gt_point[0]) + length, int(gt_point[1]) - length)
                    img = draw_line(
                        img,
                        coord_y,
                        coord_x,
                        color,
                        self.linewidth,
                    )
        img = np.array(img)
        return img


def fourier_position_embed_xy(
    xy: Float[Tensor, "num_frame batch coords"], channels: int, temperature: int = 10000
):
    device = xy.device

    assert xy.shape[-1] == 2
    assert (channels % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(channels // 4, device=device) / (channels // 4 - 1)
    omega = 1.0 / (temperature**omega)

    x = xy[:, :, :1]
    x: Float[Tensor, "num_frame batch channels_by_4"] = x * omega.reshape(1, 1, -1)

    y = xy[:, :, 1:]
    y: Float[Tensor, "num_frame batch channels_by_4"] = y * omega.reshape(1, 1, -1)

    pe: Float[Tensor, "num_frame batch channels"] = torch.cat(
        [x.sin(), x.cos(), y.sin(), y.cos()], dim=-1
    )
    pe: Float[Tensor, "num_frame batch channels_plus_two"] = torch.cat([pe, xy], dim=-1)
    return pe


def bilinear_sample2d(
    im: Float[Tensor, "batch channel height width"],
    x: Float[Tensor, "batch num_points"],
    y: Float[Tensor, "batch num_points"],
    return_inbounds=False,
):
    squeeze_output = False
    if im.ndim == 3:
        squeeze_output = True
        im = im.unsqueeze(0)
        assert x.ndim == y.ndim == 1
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)

    # x and y are each B, N
    # output is B, C, N
    B, C, H, W = list(im.shape)
    N = list(x.shape)[1]

    x = x.float()
    y = y.float()
    H_f = torch.tensor(H, dtype=torch.float32)
    W_f = torch.tensor(W, dtype=torch.float32)

    # inbound_mask = (x>-0.5).float()*(y>-0.5).float()*(x<W_f+0.5).float()*(y<H_f+0.5).float()

    max_y = (H_f - 1).int()
    max_x = (W_f - 1).int()

    x0 = torch.floor(x).int()
    x1 = x0 + 1
    y0 = torch.floor(y).int()
    y1 = y0 + 1

    x0_clip = torch.clamp(x0, 0, max_x)
    x1_clip = torch.clamp(x1, 0, max_x)
    y0_clip = torch.clamp(y0, 0, max_y)
    y1_clip = torch.clamp(y1, 0, max_y)
    dim2 = W
    dim1 = W * H

    base = torch.arange(0, B, dtype=torch.int64, device=x.device) * dim1
    base = torch.reshape(base, [B, 1]).repeat([1, N])

    base_y0 = base + y0_clip * dim2
    base_y1 = base + y1_clip * dim2

    idx_y0_x0 = base_y0 + x0_clip
    idx_y0_x1 = base_y0 + x1_clip
    idx_y1_x0 = base_y1 + x0_clip
    idx_y1_x1 = base_y1 + x1_clip

    # use the indices to lookup pixels in the flat image
    # im is B x C x H x W
    # move C out to last dim
    im_flat = (im.permute(0, 2, 3, 1)).reshape(B * H * W, C)
    i_y0_x0 = im_flat[idx_y0_x0.long()]
    i_y0_x1 = im_flat[idx_y0_x1.long()]
    i_y1_x0 = im_flat[idx_y1_x0.long()]
    i_y1_x1 = im_flat[idx_y1_x1.long()]

    # Finally calculate interpolated values.
    x0_f = x0.float()
    x1_f = x1.float()
    y0_f = y0.float()
    y1_f = y1.float()

    w_y0_x0 = ((x1_f - x) * (y1_f - y)).unsqueeze(2)
    w_y0_x1 = ((x - x0_f) * (y1_f - y)).unsqueeze(2)
    w_y1_x0 = ((x1_f - x) * (y - y0_f)).unsqueeze(2)
    w_y1_x1 = ((x - x0_f) * (y - y0_f)).unsqueeze(2)

    output = (
        w_y0_x0 * i_y0_x0 + w_y0_x1 * i_y0_x1 + w_y1_x0 * i_y1_x0 + w_y1_x1 * i_y1_x1
    )
    # output is B*N x C
    output = output.view(B, -1, C)
    output = output.permute(0, 2, 1)
    # output is B x C x N

    if return_inbounds:
        x_valid = (x > -0.5).byte() & (x < float(W_f - 0.5)).byte()
        y_valid = (y > -0.5).byte() & (y < float(H_f - 0.5)).byte()
        inbounds = (x_valid & y_valid).float()
        inbounds = inbounds.reshape(
            B, N
        )  # something seems wrong here for B>1; i'm getting an error here (or downstream if i put -1)

        if squeeze_output:
            output = output.squeeze(0)
            inbounds = inbounds.squeeze(0)

        return output, inbounds

    if squeeze_output:
        output = output.squeeze(0)

    return output  # B, C, N
