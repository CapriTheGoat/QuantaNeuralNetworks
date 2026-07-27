"""
Full Dataset Evaluation for SPAD Classification (Fixed)
"""
from pathlib import Path

import hydra
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from quanta_neural_networks.classification.classification_morphed_priority import BaselineClassifier
from quanta_neural_networks.classification.dataloader_random_morph import stochastic_spad_morph
from quanta_neural_networks.classification.dataloader import IntensityCubeSimulatedNPYMorphed

@hydra.main(
    config_path=f"../../conf",
    config_name=f"{Path(__file__).parent.name}_{Path(__file__).stem}",
    version_base="1.2",
)
def evaluate_full_dataset(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_dataset = IntensityCubeSimulatedNPYMorphed(**cfg.data.test)
    
    test_dataloader = DataLoader(
        test_dataset, 
        shuffle=False, 
        batch_size=cfg.data.batch_size, 
        num_workers=cfg.data.num_workers,
        drop_last=True
    )

    
    # 1. Instantiate the raw, uncompiled model template
    model = BaselineClassifier(**cfg.model.kwargs).to(device)

    ckpt_path = Path(cfg.model.ckpt.folder)
    print(f"Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path / f"checkpoint.pth", map_location=device)
    
    # 2. Strip any compiled "_orig_mod." prefixes from the saved file keys
    raw_state_dict = checkpoint["model"]
    clean_state_dict = {}
    for k, v in raw_state_dict.items():
        name = k.replace("_orig_mod.", "")
        clean_state_dict[name] = v
        
    # 3. Filter out transient integrator buffers by comparing against the base model template
    base_model_state = model.state_dict()
    filtered_state_dict = {
        k: v for k, v in clean_state_dict.items() if k in base_model_state
    }
    
    # 4. Load the true neural network weights strictly into the raw architecture
    model.load_state_dict(filtered_state_dict, strict=True)
    print("Core neural network weights successfully verified and locked in.")
    
    # 5. NOW compile the model for high-speed evaluation execution
    import torch._inductor.config as inductor_config
    inductor_config.layout_optimization = False
    model = torch.compile(model)
    
    model.eval()

    correct_predictions = 0
    total_samples = 0
    total_correct_400 = 0
    total_correct_1000 = 0
    total_correct_600 = 0
    total_milestone_samples = 0

    total_start_correct = 0
    total_start_samples = 0
    total_end_correct = 0
    total_end_samples = 0
    total_final_frame_correct = 0

    frame_by_frame_correct = None
    frame_by_frame_valid = None
    
    print(f"\n--- Starting Full Evaluation on {len(test_dataset)} samples ---")

    subsample_factor = 5

    with torch.no_grad(), tqdm(total=len(test_dataset), dynamic_ncols=True) as pbar:
        for batch in test_dataloader:
            label_A, label_B, cube_A, cube_B = batch

            cube_A = cube_A.to(device)
            cube_B = cube_B.to(device)
            label_A = label_A.to(device)
            label_B = label_B.to(device)

            morphed_cube, target_label = stochastic_spad_morph(cube_A, cube_B, label_A, label_B)
            T = morphed_cube.shape[3]

            target_label = target_label[:, ::subsample_factor]

            logits = model.simulate_live_camera(morphed_cube)
            logits = logits.permute(1, 2, 0).contiguous()
            
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)

            predicted_class = torch.argmax(logits, dim=1)

            valid_mask = target_label != -100
            correct_predictions += ((predicted_class == target_label) & valid_mask).sum().item()
            total_samples += valid_mask.sum().item()

            if frame_by_frame_correct is None:
                    T_sub = target_label.shape[1]
                    frame_by_frame_correct = torch.zeros(T_sub, device=device)
                    frame_by_frame_valid = torch.zeros(T_sub, device=device)
                
            frame_by_frame_correct += ((predicted_class == target_label) & valid_mask).sum(dim=0)
            frame_by_frame_valid += valid_mask.sum(dim=0)
            
            #Start Accuracy
            start_mask = valid_mask[:, :T//4]
            total_start_correct += ((predicted_class[:, :T//4] == target_label[:, :T//4]) & start_mask).sum().item()
            total_start_samples += start_mask.sum().item()

            #End Accuracy
            end_mask = valid_mask[:, -T//4:]
            total_end_correct += ((predicted_class[:, -T//4:] == target_label[:, -T//4:]) & end_mask).sum().item()
            total_end_samples += end_mask.sum().item()
                
            #Final Frame Accuracy
            total_final_frame_correct += (predicted_class[:, -1] == target_label[:, -1]).sum().item()

            
            idx_400 = 400 // subsample_factor
            idx_1000 = 1000 // subsample_factor -1
            idx_600 = 600 // subsample_factor

            prediction_400 = predicted_class[:, idx_400]
            target_400 = target_label[:, idx_400]
            
            prediction_1000 = predicted_class[:, idx_1000]
            target_1000 = target_label[:, idx_1000]

            prediction_600 = predicted_class[:, idx_600]
            target_600 = target_label[:, idx_600]

            total_correct_400 += (prediction_400 == target_400).sum().item()
            total_correct_1000 += (prediction_1000 == target_1000).sum().item()
            total_correct_600 += (prediction_600 == target_600).sum().item()
            total_milestone_samples += target_label.size(0)

            pbar.update(target_label.size(0))

    final_accuracy = (correct_predictions / total_samples) * 100
    final_acc_400 = (total_correct_400 / total_milestone_samples) * 100
    final_acc_1000 = (total_correct_1000 / total_milestone_samples) * 100
    final_acc_600 = (total_correct_600 / total_milestone_samples) * 100
    
    print(f"\n====================================")
    print(f"          FINAL RESULTS             ")
    print(f"====================================")
    print(f"Total Samples Tested: {total_samples}")
    print(f"Overall Accuracy:     {final_accuracy:.2f}%")
    print("EVALUATION MILESTONES:")
    print(f"Accuracy right before morph (Frame 400):  {final_acc_400:.1f}%")
    print(f"Accuracy right after morph (Frame 600):  {final_acc_600:.1f}%")
    print(f"Accuracy at the very end (Frame 1000):    {final_acc_1000:.1f}%")
    print(f"====================================\n")

    if frame_by_frame_correct is not None:
            import matplotlib.pyplot as plt
            
            # Calculate percentage per frame: (Correct / Total Valid) * 100
            avg_timeline = (frame_by_frame_correct / frame_by_frame_valid.clamp(min=1)) * 100
            avg_timeline = avg_timeline.cpu().numpy()
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(avg_timeline, label="Average Accuracy", color="#1f77b4", linewidth=2.5)
            
            # Draw a vertical line exactly in the middle where the morph happens
            ax.axvline(x=len(avg_timeline)//2, color="red", linestyle="--", label="Morph Point")
            
            ax.set_title(f"Average Frame-by-Frame Accuracy")
            ax.set_xlabel("Sub-sampled Frames")
            ax.set_ylabel("Accuracy (%)")
            ax.set_ylim(0, 105)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.savefig("evaluation_timeline.png", bbox_inches='tight')
            print("Saved graph to evaluation_timeline.png")
            plt.close(fig)

if __name__ == "__main__":
    evaluate_full_dataset()