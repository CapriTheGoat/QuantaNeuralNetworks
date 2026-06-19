import numpy as np
import matplotlib.pyplot as plt

# Load one random file you generated
cube = np.load("./spad_mnist_static/train/0/image_00037.npy")

print(f"Shape: {cube.shape}") 
print(f"Data Type: {cube.dtype}")
print(f"Min value: {cube.min()} | Max value: {cube.max()}")

# Sum all 1000 frames together along the Time axis (axis=2)
# This mimics a "long exposure" photograph
long_exposure = np.sum(cube, axis=2)

plt.imshow(long_exposure, cmap='gray')
plt.title("1000-Frame Summation")
plt.colorbar()
plt.show()