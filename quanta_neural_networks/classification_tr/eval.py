"""
Evaluation entrypoint for classification model.
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

from quanta_neural_networks.ssd import SSD
from quanta_neural_networks.ops.array_ops import loguniform
from quanta_neural_networks.ops.metrics import PSNR
from quanta_neural_networks.classification_tr.dataloader import TRMNISTAsynchronousDataset
from quanta_neural_networks.classification_tr.classification_tr import BaselineClassifier
from quanta_neural_networks.utils.hydra import print_and_save_cfg
from quanta_neural_networks.utils.train_utils import (
    resume_or_finetune,
    simulate_photon_cube,
)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

@hydra.main(
    config_path=f"../../conf",
    config_name=f"{Path(__file__).parent.name}_{Path(__file__).stem}",
    version_base="1.2",
)

def main (cfg):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    logger.info(f"Using device {device}")

    test_dataset = TRMNISTAsynchronousDataset(**cfg.data.test)

    
    test_dataloader = DataLoader(
        test_dataset,
        shuffle=True,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        prefetch_factor=2,
    )

    
    model = BaselineClassifier(**cfg.model.kwargs).to(device)

    for module in model.modules():
        if isinstance(module, SSD):
            module.parallel_mode = cfg.model.get("parallel_mode", False)
    
    ckpt_dir = Path(cfg.model.ckpt.folder)
    ckpt_dir.mkdir(exist_ok=True, parents=True)

    correct_predictions = 0
    total_samples = 0

    sample_idx = 0
    target_label, event_stream = test_dataset[sample_idx]
    print(f"Target Label: {target_label}")
    print(f"Total events in sample: {len(event_stream)}")
    print(f"Min y: {event_stream[:, 0].min()}, Max y: {event_stream[:, 0].max()}")
    print(f"Min x: {event_stream[:, 1].min()}, Max x: {event_stream[:, 1].max()}")
        
    print(f"\n--- Starting Full Evaluation on {len(test_dataset)} samples ---")
    
    with torch.no_grad(), tqdm(total=len(test_dataset), dynamic_ncols=True) as pbar:
        for index, batch in enumerate(test_dataloader):

            labels, events_batch = batch 
            
            for b_idx in range(len(labels)):
                single_stream = events_batch[b_idx] 
                label = labels[b_idx].item()
                
                valid_mask = single_stream[:, 2] > 0
                clean_stream = single_stream[valid_mask]

                real_max_time = clean_stream[:, 2].max().item()
                
                logits = model.simulate_live_camera(clean_stream)
                
                final_pred = logits.argmax(dim=1).item()
                
                if final_pred == label:
                    correct_predictions += 1
                total_samples += 1
                
                pbar.update(1)
                pbar.set_postfix({"Acc": f"{(correct_predictions/total_samples)*100:.1f}%"})
    
    final_accuracy = (correct_predictions / total_samples) * 100
        
    print(f"\n====================================")
    print(f"          FINAL RESULTS             ")
    print(f"====================================")
    print(f"Total Samples Tested: {total_samples}")
    print(f"Overall Accuracy:     {final_accuracy:.2f}%")
    print(f"====================================\n")
    

    

    

if __name__ == "__main__":
    main()
