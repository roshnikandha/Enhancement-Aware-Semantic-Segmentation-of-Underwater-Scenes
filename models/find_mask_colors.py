import cv2
import numpy as np

mask = cv2.imread("dataset/train_val/masks/d_r_1_.bmp")
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

print("Mask Shape:", mask.shape)

colors = np.unique(mask.reshape(-1, 3), axis=0)

print("Number of Colors:", len(colors))

print(colors)