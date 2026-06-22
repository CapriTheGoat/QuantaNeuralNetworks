import os
import numpy as np
import torch
from torchvision import datasets, transforms
from visionsim.emulate.spc import emulate_spc
from tqdm import tqdm

OUTPUT_DIR = "./data/sequences/spad_mnist_1000"
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
TEST_DIR = os.path.join(OUTPUT_DIR, "test")

T = 1000               
DCR_NOISE = 0
SPAD_FACTOR = 2.0     # Brightness factor (Lower = Harder/Noisier)

# Create the main directories
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# Create the subfolders
for i in range(10):
    os.makedirs(os.path.join(TRAIN_DIR, str(i)), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, str(i)), exist_ok=True)

print("Downloading MNIST...")
transform = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

def generate_and_save(dataset, output_base_folder, split_name):
    print(f"Generating {split_name} dataset ({len(dataset)} images)...")
    
    for index in tqdm(range(len(dataset)), desc=f"Simulating {split_name}"):
        
        image, label = dataset[index]
        image_np = image.squeeze().numpy() 
        
        clean_video_np = np.repeat(image_np[np.newaxis, :, :], T, axis=0)
        
        clean_video_np = np.clip(clean_video_np + DCR_NOISE, 0.0, 1.0)
        noisy_spad_cube = emulate_spc(clean_video_np, factor=SPAD_FACTOR)
        
        noisy_spad_cube = noisy_spad_cube.transpose(1, 2, 0)
        noisy_spad_cube = noisy_spad_cube.astype(np.uint8)
        
        save_path = os.path.join(output_base_folder, str(label), f"image_{index:05d}.npy")
        np.save(save_path, noisy_spad_cube)


generate_and_save(mnist_train, TRAIN_DIR, "Train")
generate_and_save(mnist_test, TEST_DIR, "Test")

print("\nDone")
