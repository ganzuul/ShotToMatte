import sys
print(f"Python version: {sys.version}")

import torch
print("\nPyTorch:")
print(f"Version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name()}")
    print(f"CUDA version: {torch.version.cuda}")
    # Test CUDA tensor operations
    x = torch.rand(5, 3).cuda()
    y = torch.rand(5, 3).cuda()
    print("CUDA tensor test:", torch.sum(x * y).item())

import numpy as np
print("\nNumPy:")
print(f"Version: {np.__version__}")
# Test basic numpy operation
arr = np.random.rand(5, 3)
print("NumPy test:", np.mean(arr))

import cv2
print("\nOpenCV:")
print(f"Version: {cv2.__version__}")
# Test if OpenCV can use CUDA
print("OpenCV CUDA available:", cv2.cuda.getCudaEnabledDeviceCount() > 0)

import tqdm
print("\ntqdm:")
print(f"Version: {tqdm.__version__}")
# Quick tqdm test
for _ in tqdm.tqdm(range(3), desc="tqdm test"):
    pass 