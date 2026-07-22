import os

# Folder paths
image_folder = "dataset/train_val/images"
mask_folder = "dataset/train_val/masks"

# Get all image and mask filenames
image_files = sorted(os.listdir(image_folder))
mask_files = sorted(os.listdir(mask_folder))

print("Total Images:", len(image_files))
print("Total Masks:", len(mask_files))

print("\nFirst 10 image-mask pairs:\n")

for image_name, mask_name in zip(image_files[:10], mask_files[:10]):
    print(f"{image_name}  --->  {mask_name}")