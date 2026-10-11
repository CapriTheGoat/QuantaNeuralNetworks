"""
Evaluation entrypoint for classification model.
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path

import hydra
import time
from hydra.utils import to_absolute_path
import numpy as np
import torch
from einops import rearrange
from loguru import logger
from piq import ssim
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.optim as optim
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from quanta_neural_networks.ssd_tr import SSD
from quanta_neural_networks.ops.array_ops import loguniform
from quanta_neural_networks.ops.metrics import PSNR
from quanta_neural_networks.classification_tr.dataloader import TRMNISTAsynchronousDataset
from quanta_neural_networks.classification_tr.classification_tr import BaselineClassifier
from quanta_neural_networks.utils.hydra import print_and_save_cfg
from quanta_neural_networks.utils.train_utils import (
    resume_or_finetune,
    simulate_photon_cube,
)
def stream_to_delta_t_cube(
    event_stream: torch.Tensor,
    spatial_size=(28, 28),
    num_frames=100,
    max_time=1e-3,
    coordinate_base=1,
):
    """Return interarrival times and binary masks: [1,H,W,T,K]."""
    if event_stream.dim() == 3:
        if event_stream.shape[0] != 1:
            raise ValueError("Set data.batch_size=1 for this converter.")
        event_stream = event_stream[0]

    if event_stream.ndim != 2 or event_stream.shape[1] != 3:
        raise ValueError("Expected events with shape [N,3]: (y,x,t).")
    if coordinate_base not in (0, 1):
        raise ValueError("coordinate_base must be 0 or 1.")
    if num_frames < 1 or max_time <= 0:
        raise ValueError("num_frames and max_time must be positive.")

    H, W = spatial_size
    device = event_stream.device
    events = event_stream.to(torch.float64)

    if not torch.isfinite(events).all():
        raise ValueError("Events must contain finite coordinates and times.")

    events = events[torch.argsort(events[:, 2], stable=True)]

    xy = events[:, :2] - coordinate_base
    if (xy != xy.round()).any():
        raise ValueError("Pixel coordinates must be integers.")

    y, x = xy[:, 0].long(), xy[:, 1].long()
    t = events[:, 2]

    if ((y < 0) | (y >= H) | (x < 0) | (x >= W)).any():
        raise ValueError("Check coordinate_base and sensor dimensions.")
    boundary_tolerance = 1e-9 

    too_early = t < -boundary_tolerance
    too_late = t > max_time + boundary_tolerance

    if (too_early | too_late).any():
        raise ValueError(
            "Timestamp range exceeds boundary tolerance: "
            f"t_min={t.min().item():.17g}, "
            f"t_max={t.max().item():.17g}, "
            f"max_time={max_time:.17g}, "
            f"too_early={too_early.sum().item()}, "
            f"too_late={too_late.sum().item()}"
        )

    t = t.clamp(min=0.0, max=max_time)

    time_per_frame = max_time / num_frames
    frame_indices = (t / time_per_frame).floor().long()
    frame_indices = frame_indices.clamp(max=num_frames - 1)

    # Counts are used ONLY to allocate enough event slots.
    cell = (y * W + x) * num_frames + frame_indices
    occupancy = torch.bincount(cell, minlength=H * W * num_frames)
    K = max(1, int(occupancy.max().item()))

    delta_t_cube = torch.zeros(
        (H, W, num_frames, K), device=device, dtype=torch.float64
    )
    photon_mask = torch.zeros_like(
        delta_t_cube, dtype=torch.float32
    )
    order = torch.argsort(cell, stable=True)

    ordered_cells = cell[order]
    ordered_t = t[order]
    ordered_pixels = ordered_cells // num_frames

    # Previous arrival at the same pixel, including across bin boundaries.
    previous_t = torch.zeros_like(ordered_t)
    previous_t[1:] = torch.where(
        ordered_pixels[1:] == ordered_pixels[:-1],
        ordered_t[:-1],
        0.0,
    )

    interarrival_times = ordered_t - previous_t

    # Position of each photon within its pixel/bin group.
    starts = occupancy.cumsum(dim=0) - occupancy
    slots = (
        torch.arange(len(events), device=device)
        - starts[ordered_cells]
    )

    delta_t_cube.view(-1, K)[ordered_cells, slots] = interarrival_times
    photon_mask.view(-1, K)[ordered_cells, slots] = 1.0

    return (
        delta_t_cube.unsqueeze(0),
        photon_mask.unsqueeze(0),
        time_per_frame,
    )

def camera_stream(
    dataloader,
    packet_bins,
    num_raw_bins=100,
    exposure=1e-3,
    wall_start=None,
):
    """
    Present successive dataset samples as a continuous camera stream.

    Yield only NEW events, packed for the existing integrator.
    Labels are returned solely for evaluation.
    """
    if (
        packet_bins < 1
        or num_raw_bins < packet_bins
        or num_raw_bins % packet_bins != 0
    ):
        raise ValueError(
            "num_raw_bins must be a positive multiple of packet_bins."
        )

    bin_width = exposure / num_raw_bins
    packet_duration = packet_bins * bin_width

    # Absolute timestamp of the last photon at every pixel.
    previous_times = torch.zeros(
        (1, 28, 28),
        dtype=torch.float64,
    )

    for clip_index, (target_label, event_stream) in enumerate(dataloader):
        previous_times.zero_()
        if event_stream.ndim != 3 or event_stream.shape[0] != 1:
            raise ValueError("Camera evaluation requires batch_size=1.")

        events = event_stream[0].to(
            device="cpu",
            dtype=torch.float64,
        ).clone()

        timestamps = events[:, 2]
        tolerance = 1e-9

        if not torch.isfinite(timestamps).all():
            raise ValueError(f"Non-finite timestamps in clip {clip_index}.")

        if (
            (timestamps < -tolerance)
            | (timestamps > exposure + tolerance)
        ).any():
            raise ValueError(
                f"Timestamp range mismatch in clip {clip_index}."
            )

        events[:, 2] = timestamps.clamp(0.0, exposure)
        events = events[
            torch.argsort(events[:, 2], stable=True)
        ]

        # Place this recording on one continuous camera clock.
        offset = 0.0
        events[:, 2] += offset
        timestamps = events[:, 2].contiguous()

        cursor = 0

        for first_bin in range(0, num_raw_bins, packet_bins):
            end_bin = first_bin + packet_bins
            last_packet = end_bin == num_raw_bins

            start_time = offset + first_bin * bin_width
            end_time = offset + (
                exposure if last_packet else end_bin * bin_width
            )

            stop = torch.searchsorted(
                timestamps,
                timestamps.new_tensor(end_time),
                right=last_packet,
            ).item()

            # Each event is consumed once.
            packet = events[cursor:stop].clone()
            cursor = stop

            # Optional pacing against the actual wall clock.
            # If processing is late, retain the queued packet.
            if wall_start is not None:
                replay_time = clip_index * exposure + end_time
                delay = wall_start + replay_time - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)

            # Local timestamps are used only for assigning packet bins.
            packet[:, 2] -= start_time

            if len(packet) == 0:
                shape = (1, 28, 28, packet_bins, 1)
                delta_cube = torch.zeros(shape, dtype=torch.float64)
                photon_mask = torch.zeros(shape, dtype=torch.float32)
            else:
                delta_cube, photon_mask, _ = stream_to_delta_t_cube(
                    packet,
                    num_frames=packet_bins,
                    max_time=packet_duration,
                )
                delta_cube = delta_cube.double()

            # The converter starts each pixel's clock at local zero.
            # Correct its FIRST interval using the preceding packet's
            # last photon timestamp.
            occupied_bins = photon_mask.bool().any(dim=-1)
            active_pixels = occupied_bins.any(dim=-1)
            first_event_bin = occupied_bins.long().argmax(dim=-1)

            b, y, x = active_pixels.nonzero(as_tuple=True)

            delta_cube[
                b, y, x, first_event_bin[b, y, x], 0
            ] += start_time - previous_times[b, y, x]

            # Sum of the corrected intervals advances each pixel to
            # its newest absolute photon timestamp.
            previous_times += (
                delta_cube * photon_mask
            ).sum(dim=(-2, -1))

            yield (
                delta_cube,
                photon_mask,
                bin_width,
                end_time,
                target_label,
                last_packet,
            )

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

def save_evaluation_image(
    confusion_matrix,
    total_samples,
    correct_predictions,
    final_accuracy,
    mean_loss,
    correct_readouts,
    total_readouts,
    output_path,
):
    cm = np.asarray(confusion_matrix, dtype=np.int64)
    support = cm.sum(axis=1)

    row_percent = np.divide(
        100.0 * cm,
        support[:, None],
        out=np.zeros_like(cm, dtype=float),
        where=support[:, None] > 0,
    )
    recalls = row_percent.diagonal()

    readout_accuracy = (
        f"{100.0 * correct_readouts / total_readouts:.2f}%"
        if total_readouts else "N/A"
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = Figure(figsize=(15, 9), facecolor="#f4f7fb")
    FigureCanvasAgg(fig)

    try:
        fig.text(
            0.04, 0.955,
            "TR-MNIST | Camera evaluation",
            fontsize=22, weight="bold", color="#142c46",
        )
        fig.text(
            0.04, 0.925,
            "Final recording predictions and accuracy across all camera readouts",
            fontsize=11, color="#50657c",
        )

        # Summary metrics.
        metrics = [
            ("SAMPLES TESTED", f"{total_samples:,}"),
            ("CORRECT FINAL PREDICTIONS", f"{correct_predictions:,}"),
            ("FINAL ACCURACY", f"{final_accuracy:.2f}%"),
            ("ALL-READOUT ACCURACY", readout_accuracy),
            ("FINAL MEAN CROSS-ENTROPY", f"{mean_loss:.4f}"),
        ]

        for i, (label, value) in enumerate(metrics):
            card = fig.add_axes(
                [0.04 + i * 0.186, 0.80, 0.174, 0.105]
            )
            card.set_facecolor("white")
            card.set_xticks([])
            card.set_yticks([])

            for spine in card.spines.values():
                spine.set_color("#e2e8f0")

            card.text(
                0.07, 0.72, label,
                fontsize=8.5, color="#50657c",
            )
            card.text(
                0.07, 0.20, value,
                fontsize=25, weight="bold", color="#142c46",
            )

        # Confusion matrix: printed values are counts.
        # Color represents the percentage within each true-digit row.
        ax_cm = fig.add_axes([0.055, 0.17, 0.405, 0.57])
        heatmap = ax_cm.imshow(
            row_percent, cmap="Blues", vmin=0, vmax=100
        )
        ax_cm.set_title(
            "Confusion matrix | Final readout",
            loc="left", pad=18,
            fontsize=13, weight="bold", color="#142c46",
        )
        ax_cm.set_xticks(range(10))
        ax_cm.set_yticks(range(10))
        ax_cm.set_xlabel("Predicted digit", labelpad=10)
        ax_cm.set_ylabel("True digit", labelpad=10)

        ax_cm.set_xticks(np.arange(-0.5, 10, 1), minor=True)
        ax_cm.set_yticks(np.arange(-0.5, 10, 1), minor=True)
        ax_cm.grid(which="minor", color="white", linewidth=1)
        ax_cm.tick_params(which="both", length=0)

        for spine in ax_cm.spines.values():
            spine.set_visible(False)

        for true_digit in range(10):
            for predicted_digit in range(10):
                count = cm[true_digit, predicted_digit]
                color = (
                    "white"
                    if row_percent[true_digit, predicted_digit] > 55
                    else "#142c46"
                )
                ax_cm.text(
                    predicted_digit, true_digit, str(count),
                    ha="center", va="center",
                    fontsize=9, color=color,
                )

        colorbar_ax = fig.add_axes([0.465, 0.17, 0.013, 0.57])
        colorbar = fig.colorbar(heatmap, cax=colorbar_ax)
        colorbar.set_label(
            "Percentage within each true-digit row (%)",
            fontsize=9,
        )
        colorbar.outline.set_visible(False)

        # Per-digit recall, including correct/total sample counts.
        ax_recall = fig.add_axes([0.57, 0.17, 0.39, 0.57])
        ax_recall.barh(
            range(10), recalls, height=0.58, color="#188a84"
        )
        ax_recall.set_title(
            "Per-digit recall | Final readout",
            loc="left", pad=18,
            fontsize=13, weight="bold", color="#142c46",
        )
        ax_recall.set_yticks(range(10))
        ax_recall.set_ylim(9.5, -0.5)
        ax_recall.set_xlim(0, 160)
        ax_recall.set_xticks([0, 25, 50, 75, 100])
        ax_recall.set_xlabel("Recall (%)", labelpad=10)
        ax_recall.set_ylabel("Digit")
        ax_recall.set_axisbelow(True)
        ax_recall.grid(
            axis="x", color="#e2e8f0", linewidth=0.8
        )
        ax_recall.tick_params(length=0)

        for spine in ax_recall.spines.values():
            spine.set_visible(False)

        for digit in range(10):
            label = (
                f"{recalls[digit]:.2f}%  "
                f"({cm[digit, digit]}/{support[digit]})"
                if support[digit]
                else "N/A (no samples)"
            )
            ax_recall.text(
                104, digit, label,
                va="center", fontsize=9, color="#142c46",
            )

        fig.text(
            0.04, 0.065,
            "Matrix labels are counts; color shows row percentages. "
            "Final metrics use one prediction per recording.",
            fontsize=10, color="#50657c",
        )
        fig.text(
            0.04, 0.035,
            f"Camera readouts: {total_readouts:,} | "
            f"Correct readouts: {correct_readouts:,}. "
            "All-readout accuracy includes intermediate and final predictions.",
            fontsize=10, color="#50657c",
        )

        fig.savefig(
            output_path,
            dpi=200,
            facecolor=fig.get_facecolor(),
        )
    finally:
        fig.clear()

    return output_path

@hydra.main(
    config_path=f"../../conf",
    config_name=f"{Path(__file__).parent.name}_{Path(__file__).stem}",
    version_base="1.2",
)

def main (cfg):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    logger.info(f"Using device {device}")

    test_dataset = TRMNISTAsynchronousDataset(**cfg.data.test)

    num_workers = int(cfg.data.num_workers)

    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        generator=torch.Generator().manual_seed(0),
        batch_size=1,
        num_workers=0,
        pin_memory=False,
    )

    
    model = BaselineClassifier(**cfg.model.kwargs).to(device)

    for module in model.modules():
        if isinstance(module, SSD):
            module.parallel_mode = cfg.model.get("parallel_mode", False)
    
    ckpt_dir = Path(to_absolute_path(str(cfg.model.ckpt.folder)))
    ckpt_path = ckpt_dir / "checkpoint.pth"

    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}"
        )

    checkpoint = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )

    # Your training script saves model weights under the "model" key.
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    del checkpoint

    logger.info(f"Loaded checkpoint: {ckpt_path}")


    criterion = nn.CrossEntropyLoss(reduction="sum")

    correct_predictions = 0
    total_samples = 0
    total_loss = 0.0
    confusion_matrix = torch.zeros((10, 10), dtype=torch.long)

    num_raw_bins = 100
    exposure = 1e-3
    packet_bins = int(model.subsampling)

    first_packet_of_recording = True

    correct_readouts = 0
    total_readouts = 0

    # True attempts to deliver packets at their camera timestamps.
    # False runs the same stateful stream without intentional waiting.
    pace_replay = True
    wall_start = time.perf_counter()

    stream = camera_stream(
        test_dataloader,
        packet_bins=packet_bins,
        num_raw_bins=num_raw_bins,
        exposure=exposure,
        wall_start=wall_start if pace_replay else None,
    )

    with torch.no_grad(), tqdm(
        total=len(test_dataset),
        dynamic_ncols=True,
        desc="Camera evaluation",
    ) as pbar:

        for (
            delta_cube,
            photon_mask,
            time_per_frame,
            camera_time,
            target_label,
            last_packet,
        ) in stream:

            target_label = target_label.to(
                device=device,
                dtype=torch.long,
            ).reshape(-1)

            if first_packet_of_recording:
                model.reset_camera(
                    history_readouts=num_raw_bins // packet_bins
            )

            logits = model.forward_camera(
                delta_cube.to(device),
                photon_mask.to(device),
                time_per_frame,
                current_time=camera_time,
            )
            first_packet_of_recording = bool(last_packet)

            if not torch.isfinite(logits).all():
                raise RuntimeError(
                    f"Non-finite logits at camera time {camera_time}."
                )

            predicted_class = logits.argmax(dim=1)
            correct = (
                predicted_class == target_label
            ).sum().item()

            correct_readouts += correct
            total_readouts += target_label.numel()

            # Display the first few live predictions.
            if total_readouts <= 10:
                logger.info(
                    f"Camera time: {camera_time * 1e3:.3f} ms | "
                    f"Prediction: {predicted_class.item()}"
                )

            # Keep your existing final-results report:
            # score once at the end of each displayed recording.
            if last_packet:
                total_loss += criterion(logits, target_label).item()
                correct_predictions += correct
                total_samples += target_label.numel()

                pairs = (
                    10 * target_label + predicted_class
                ).cpu()

                confusion_matrix += torch.bincount(
                    pairs,
                    minlength=100,
                ).reshape(10, 10)

                pbar.update(target_label.numel())
                pbar.set_postfix(
                    accuracy=(
                        f"{100 * correct_predictions / total_samples:.2f}%"
                    )
                )

    if total_readouts:
        print(
            "\nAccuracy across all camera readouts: "
            f"{100 * correct_readouts / total_readouts:.2f}%"
        )

    if total_samples == 0:
        raise ValueError("The test dataset is empty.")

    final_accuracy = 100.0 * correct_predictions / total_samples
    mean_loss = total_loss / total_samples

    print("\nFINAL RESULTS")
    print(f"Total samples tested: {total_samples}")
    print(f"Correct predictions:  {correct_predictions}")
    print(f"Overall accuracy:     {final_accuracy:.2f}%")
    print(f"Mean cross-entropy:   {mean_loss:.4f}")

    print("\nPer-digit recall:")
    for digit in range(10):
        digit_total = confusion_matrix[digit].sum().item()
        digit_correct = confusion_matrix[digit, digit].item()

        if digit_total > 0:
            recall = 100.0 * digit_correct / digit_total
            print(
                f"Digit {digit}: {recall:.2f}% "
                f"({digit_correct}/{digit_total})"
            )
        else:
            print(f"Digit {digit}: N/A — no test samples")

    print(
        "\nConfusion matrix "
        "(rows=true, columns=predicted; digits 0–9):"
    )
    print(confusion_matrix.numpy())
    image_path = ckpt_dir / (
        f"evaluation_results_{time.strftime('%Y%m%d_%H%M%S')}.png"
    )

    save_evaluation_image(
        confusion_matrix=confusion_matrix.detach().cpu().numpy(),
        total_samples=total_samples,
        correct_predictions=correct_predictions,
        final_accuracy=final_accuracy,
        mean_loss=mean_loss,
        correct_readouts=correct_readouts,
        total_readouts=total_readouts,
        output_path=image_path,
    )

    logger.info(f"Saved evaluation image: {image_path}")

if __name__ == "__main__":
    main()
