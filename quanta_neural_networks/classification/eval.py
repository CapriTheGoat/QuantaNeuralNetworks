"""
Evaluation and visualization script for SPAD Classification
"""
from pathlib import Path

import hydra
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
import matplotlib.pyplot as plt
from pathlib import Path
from quanta_neural_networks.classification.classification import BaselineClassifier
from quanta_neural_networks.classification.dataloader import IntensityCubeSimulatedNPY

@hydra.main(
    config_path=f"../../conf",
    config_name=f"{Path(__file__).parent.name}_{Path(__file__).stem}",
    version_base="1.2",
)

def evaluate_single_sample(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    val_dataset = IntensityCubeSimulatedNPY(**cfg.data.val)
   

    model = BaselineClassifier(**cfg.model.kwargs).to(device)

    ckpt_path = Path(cfg.model.ckpt.folder)
    print(f"Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path / f"checkpoint.pth", map_location=device)
    model.load_state_dict(checkpoint["model"], strict=False)
    
    model.eval()

    target_label, photon_cube, intensity_ll = val_dataset[0] 
    
    photon_cube_batch = photon_cube.to(device)

    with torch.no_grad():
        logits = model(photon_cube_batch)
        predicted_class = torch.argmax(logits, dim=1).item()

    print(f"\n--- Results ---")
    print(f"True Digit:      {target_label.item()}")
    print(f"Model Predicted: {predicted_class}")
    print(f"Confidence:      {torch.softmax(logits, dim=1)[0, predicted_class]:.2%}")

    noisy_2d_image = photon_cube.sum(dim=-1).squeeze(0).cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.title(f"SPAD Input\nTrue: {target_label.item()} | Predicted: {predicted_class}")
    plt.imshow(noisy_2d_image, cmap='gray')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    evaluate_single_sample()
