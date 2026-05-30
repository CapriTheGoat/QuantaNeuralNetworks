"""
Full Dataset Evaluation for SPAD Classification
"""
from pathlib import Path

import hydra
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from quanta_neural_networks.classification.classification_morphed import BaselineClassifier
from quanta_neural_networks.classification.dataloader import IntensityCubeSimulatedNPYMorphed

@hydra.main(
    config_path=f"../../conf",
    config_name=f"{Path(__file__).parent.name}_{Path(__file__).stem}",
    version_base="1.2",
)
def evaluate_full_dataset(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_dataset = IntensityCubeSimulatedNPYMorphed(**cfg.data.val)
    
    test_dataloader = DataLoader(
        test_dataset, 
        shuffle=False, # No need to shuffle for evaluation
        batch_size=cfg.data.batch_size, 
        num_workers=cfg.data.num_workers
    )

    model = BaselineClassifier(**cfg.model.kwargs).to(device)
    ckpt_path = Path(cfg.model.ckpt.folder)
    ckpt_path.mkdir(exist_ok=True, parents=True)
    print(f"Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path / f"checkpoint.pth", map_location=device)
    model.load_state_dict(checkpoint["model"], strict=False)
    
    model.eval()

    correct_predictions = 0
    total_samples = 0
    total_correct_20 = 0
    total_correct_100 = 0
    total_milestone_samples = 0
    
    print(f"\n--- Starting Full Evaluation on {len(test_dataset)} samples ---")

    with torch.no_grad(), tqdm(total=len(test_dataset), dynamic_ncols=True) as pbar:
        for batch in test_dataloader:
            target_label, photon_cube, intensity_ll = batch
            
            photon_cube = photon_cube.to(device)
            target_label = target_label.to(device)

            logits = model.simulate_live_camera(photon_cube)

            logits = logits.permute(1, 2, 0).contiguous()
            
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)

            predicted_class = torch.argmax(logits, dim=1)

            valid_mask = target_label != -100
            correct_predictions += ((predicted_class == target_label) & valid_mask).sum().item()
            total_samples += valid_mask.sum().item()

            subsample_factor = 4

            idx_20 = (20 // subsample_factor) - 1
            idx_100 = (100 // subsample_factor) - 1

            prediction_20 = predicted_class[:, idx_20]
            target_20 = target_label[:, idx_20]
            
            prediction_100 = predicted_class[:, idx_100]
            target_100 = target_label[:, idx_100]

            correct_20 = (prediction_20 == target_20).sum().item()
            correct_100 = (prediction_100 == target_100).sum().item()

            total_correct_20 += correct_20
            total_correct_100 += correct_100
            total_milestone_samples += target_label.size(0)

            pbar.update(target_label.size(0))

    final_accuracy = (correct_predictions / total_samples) * 100

    final_acc_20 = (total_correct_20 / total_milestone_samples) * 100
    final_acc_100 = (total_correct_100 / total_milestone_samples) * 100

    
    print(f"\n====================================")
    print(f"          FINAL RESULTS             ")
    print(f"====================================")
    print(f"Total Samples Tested: {total_samples}")
    print(f"Overall Accuracy:     {final_accuracy:.2f}%")
    print("EVALUATION MILESTONES:")
    print(f"Accuracy right before morph (Frame 20):  {final_acc_20:.1f}%")
    print(f"Accuracy at the very end (Frame 100):    {final_acc_100:.1f}%")
    print(f"====================================\n")

if __name__ == "__main__":
    evaluate_full_dataset()
