"""
Full Dataset Evaluation for SPAD Classification
"""
from pathlib import Path

import hydra
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from quanta_neural_networks.classification.classification import BaselineClassifier
from quanta_neural_networks.classification.dataloader import IntensityCubeSimulatedNPY

import json
from datetime import datetime

import numpy as np
from torch.nn import functional as F

import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

def save_results(cm, mean_ce, output_dir, checkpoint_file):
    """cm[true_label, predicted_label] contains recording counts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    support = cm.sum(axis=1)
    correct_per_digit = np.diag(cm)
    total = int(support.sum())

    if total == 0:
        raise ValueError("No recordings were evaluated.")

    correct = int(correct_per_digit.sum())
    accuracy = 100.0 * correct / total

    percentages = np.divide(
        100.0 * cm,
        support[:, None],
        out=np.full(cm.shape, np.nan, dtype=float),
        where=support[:, None] > 0,
    )
    recall = np.diag(percentages)

    metrics = {
        "model": "QIS QNN",
        "checkpoint": str(checkpoint_file),
        "prediction_scope": "one returned prediction per recording",
        "total_samples": total,
        "correct_predictions": correct,
        "accuracy_percent": accuracy,
        "mean_cross_entropy": mean_ce,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_axes": {
            "rows": "true",
            "columns": "predicted",
        },
        "per_digit": [
            {
                "digit": digit,
                "total": int(support[digit]),
                "correct": int(correct_per_digit[digit]),
                "recall_percent": (
                    float(recall[digit]) if support[digit] else None
                ),
            }
            for digit in range(10)
        ],
    }

    with (output_dir / "qis_metrics.json").open("w") as file:
        json.dump(metrics, file, indent=2, allow_nan=False)

    np.savetxt(
        output_dir / "qis_confusion_matrix.csv",
        np.column_stack((np.arange(10), cm)),
        delimiter=",",
        fmt="%d",
        header="true_label," + ",".join(
            f"pred_{i}" for i in range(10)
        ),
        comments="",
    )

    fig, axes = plt.subplots(
        2, 2,
        figsize=(14, 10),
        constrained_layout=True,
    )
    fig.suptitle(
        "QIS QNN — classification results",
        fontsize=18,
    )

    panels = [
        (
            axes[0, 0],
            cm,
            "Confusion matrix: counts",
            False,
        ),
        (
            axes[0, 1],
            percentages,
            "Confusion matrix: row percentages",
            True,
        ),
    ]

    for ax, values, title, normalized in panels:
        upper = (
            100.0 if normalized
            else max(float(cm.max()), 1.0)
        )

        image = ax.imshow(
            values,
            cmap="Blues",
            vmin=0,
            vmax=upper,
        )

        ax.set_title(title)
        ax.set_xlabel("Predicted digit")
        ax.set_ylabel("True digit")
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))

        fig.colorbar(
            image,
            ax=ax,
            fraction=0.046,
            pad=0.04,
            label="Percent" if normalized else "Recordings",
        )

        for row in range(10):
            for col in range(10):
                value = values[row, col]

                if np.isnan(value):
                    label = "N/A"
                elif normalized:
                    label = f"{value:.1f}"
                else:
                    label = str(int(value))

                ax.text(
                    col, row, label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=(
                        "white" if value > upper / 2
                        else "black"
                    ),
                )

    ax = axes[1, 0]
    ax.bar(
        range(10),
        np.nan_to_num(recall, nan=0.0),
        color="#2563eb",
    )
    ax.set_title("Per-digit recall")
    ax.set_xlabel("True digit")
    ax.set_ylabel("Recall (%)")
    ax.set_xticks(range(10))
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.2)

    for digit, value in enumerate(recall):
        ax.text(
            digit,
            value + 2 if np.isfinite(value) else 2,
            f"{value:.1f}%" if np.isfinite(value) else "N/A",
            ha="center",
            fontsize=9,
        )

    ax = axes[1, 1]
    ax.axis("off")
    ax.set_title("Overall results", loc="left")
    ax.text(
        0.02, 0.92,
        f"Recordings evaluated: {total:,}\n"
        f"Correct predictions: {correct:,}\n"
        f"Incorrect predictions: {total - correct:,}\n\n"
        f"Overall accuracy: {accuracy:.2f}%\n"
        f"Mean cross-entropy: {mean_ce:.4f}\n\n"
        "One returned prediction per recording.\n"
        "Recall = correct / total for each true digit.\n"
        "N/A = no recordings for that digit.",
        transform=ax.transAxes,
        va="top",
        fontsize=12,
        linespacing=1.6,
    )

    fig.savefig(
        output_dir / "qis_results.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "qis_results.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    # Print the same main statistics as the TR evaluation.
    print("\nFINAL RESULTS")
    print(f"Total samples tested: {total}")
    print(f"Correct predictions:  {correct}")
    print(f"Overall accuracy:     {accuracy:.2f}%")
    print(f"Mean cross-entropy:   {mean_ce:.4f}")

    print("\nPer-digit recall:")
    for item in metrics["per_digit"]:
        value = item["recall_percent"]
        label = "N/A" if value is None else f"{value:.2f}%"
        print(
            f"Digit {item['digit']}: {label} "
            f"({item['correct']}/{item['total']})"
        )

    print("\nConfusion matrix (rows=true, columns=predicted):")
    print(cm)
    print(f"\nResults saved to: {output_dir.resolve()}")

@hydra.main(
    config_path=f"../../conf",
    config_name=f"{Path(__file__).parent.name}_{Path(__file__).stem}",
    version_base="1.2",
)
def evaluate_full_dataset(cfg):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device {device}")

    test_dataset = IntensityCubeSimulatedNPY(**cfg.data.test, load_intensity=False,)

    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )

    checkpoint_file = (
        Path(cfg.model.ckpt.folder).expanduser()
        / "checkpoint.pth"
    ).resolve()

    if not checkpoint_file.is_file():
        raise FileNotFoundError(checkpoint_file)

    model = BaselineClassifier(**cfg.model.kwargs).to(device)

    print(f"Loading checkpoint from {checkpoint_file}...")
    checkpoint = torch.load(
        checkpoint_file,
        map_location="cpu",
        # Only use False for your own trusted training checkpoint.
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model"], strict=False)
    del checkpoint

    model.eval()
    

    confusion_matrix = torch.zeros(
        (10, 10),
        dtype=torch.long,
        device=device,
    )
    correct_predictions = 0
    total_samples = 0
    loss_sum = 0.0

    print(
        f"\n--- Evaluating {len(test_dataset)} recordings ---"
    )

    with torch.no_grad(), tqdm(
        total=len(test_dataset),
        dynamic_ncols=True,
    ) as pbar:
        for target_label, photon_cube, intensity_ll in test_dataloader:
            photon_cube = photon_cube.to(device)
            target_label = target_label.to(
                device=device,
                dtype=torch.long,
            ).reshape(-1)

            logits = model.simulate_live_camera(photon_cube)

            if logits.ndim == 1:
                logits = logits.unsqueeze(0)

            expected_shape = (target_label.numel(), 10)

            if tuple(logits.shape) != expected_shape:
                raise ValueError(
                    "Expected one logit vector per recording: "
                    f"{expected_shape}; got {tuple(logits.shape)}."
                )

            if not torch.isfinite(logits).all():
                raise ValueError(
                    "The model returned non-finite logits."
                )

            predicted_class = logits.argmax(dim=1)
            batch_size = target_label.numel()

            correct_predictions += (
                predicted_class == target_label
            ).sum().item()
            total_samples += batch_size

            loss_sum += F.cross_entropy(
                logits.float(),
                target_label,
                reduction="sum",
            ).item()

            confusion_matrix += torch.bincount(
                10 * target_label + predicted_class,
                minlength=100,
            ).reshape(10, 10)

            pbar.update(batch_size)
            pbar.set_postfix(
                accuracy=(
                    f"{100.0 * correct_predictions / total_samples:.2f}%"
                )
            )

    if total_samples == 0:
        raise ValueError("The evaluation dataset is empty.")

    output_dir = (
        checkpoint_file.parent
        / "evaluation_qis"
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )

    save_results(
        confusion_matrix.cpu().numpy(),
        loss_sum / total_samples,
        output_dir,
        checkpoint_file,
    )
if __name__ == "__main__":
    evaluate_full_dataset()
