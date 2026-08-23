import os
import numpy as np
import torch
from torchvision import datasets, transforms
from visionsim.emulate.spc_dcr import emulate_spc
from tqdm import tqdm

# Import your new advanced functions
# from visionsim.emulate.spc_dcr import emulate_spc_advanced, generate_dcr_map

OUTPUT_DIR = "./data/sequences/spad_mnist_1000_dcr"
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
TEST_DIR = os.path.join(OUTPUT_DIR, "test")

# --- SPAD Sensor Parameters ---
T = 1000               
SPAD_FACTOR = 2.0      # Brightness factor (Lower = Harder/Noisier)

# Advanced noise parameters
DCR_BASE_RATE = 0.001       # Baseline dark counts for all pixels
HOT_PIXEL_FRACTION = 0.01   # 1% of pixels are defective
HOT_PIXEL_MULT = 50.0       # Defective pixels fire 50x more often
AFTERPULSE_PROB = 0.02      # 2% chance of a photon echoing into the next frame

# Create the main directories
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

for i in range(10):
    os.makedirs(os.path.join(TRAIN_DIR, str(i)), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, str(i)), exist_ok=True)

print("Downloading MNIST...")
transform = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Generate a single static defect map for the "sensor"
rng = np.random.default_rng(42)  # Seeded for reproducible sensor defects
sensor_shape = (28, 28)
dcr_map = np.full(sensor_shape, DCR_BASE_RATE, dtype=np.float32)
hot_mask = rng.random(sensor_shape) < HOT_PIXEL_FRACTION
dcr_map[hot_mask] *= HOT_PIXEL_MULT


def generate_and_save(dataset, output_base_folder, split_name):
    print(f"Generating {split_name} dataset ({len(dataset)} images)...")
    
    # We only need one RNG instance for the whole loop
    sim_rng = np.random.default_rng()
    
    for index in tqdm(range(len(dataset)), desc=f"Simulating {split_name}"):
        
        image, label = dataset[index]
        image_np = image.squeeze().numpy() 
        
        # Allocate the final video cube in memory [T, H, W]
        noisy_spad_cube = np.zeros((T, 28, 28), dtype=np.uint8)
        prev_frame = None
        
        # Simulate sequentially to allow afterpulse propagation
        for t in range(T):
            frame = emulate_spc(
                img=image_np, 
                prev_binary_frame=prev_frame,
                factor=SPAD_FACTOR,
                dcr_map=dcr_map,
                afterpulse_prob=AFTERPULSE_PROB,
                rng=sim_rng
            )
            noisy_spad_cube[t] = frame
            prev_frame = frame
            
        # Transpose from [T, H, W] -> [H, W, T] to match your original format
        noisy_spad_cube = noisy_spad_cube.transpose(1, 2, 0)
        
        save_path = os.path.join(output_base_folder, str(label), f"image_{index:05d}.npy")
        np.save(save_path, noisy_spad_cube)

generate_and_save(mnist_train, TRAIN_DIR, "Train")
generate_and_save(mnist_test, TEST_DIR, "Test")

print("\nDone")
