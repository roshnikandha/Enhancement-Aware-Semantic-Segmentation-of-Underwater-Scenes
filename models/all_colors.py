import os
import cv2
import numpy as np

mask_folder = "dataset/train_val/masks"

all_colors = set()

for filename in os.listdir(mask_folder):

    path = os.path.join(mask_folder, filename)

    mask = cv2.imread(path)

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    colors = np.unique(mask.reshape(-1,3), axis=0)

    for c in colors:
        all_colors.add(tuple(int(x) for x in c))

print("Total Unique Colors:", len(all_colors))
print()

for color in sorted(all_colors):
    print(color)