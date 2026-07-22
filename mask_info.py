import cv2
import numpy as np

# Read the mask
mask = cv2.imread("dataset/train_val/masks/d_r_1_.bmp")

print("Mask Shape:", mask.shape)
print("Data Type:", mask.dtype)

# Find all unique colors in the mask
pixels = mask.reshape(-1, 3)
unique_colors = np.unique(pixels, axis=0)

print("Number of unique colors:", len(unique_colors))
print("Unique colors:")
print(unique_colors)