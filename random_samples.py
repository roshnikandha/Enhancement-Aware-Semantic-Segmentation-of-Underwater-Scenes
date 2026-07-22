import os
import random
import cv2
import matplotlib.pyplot as plt

image_folder = "dataset/train_val/images"
mask_folder = "dataset/train_val/masks"

# Get image filenames
image_files = os.listdir(image_folder)

# Pick 4 random images
random_images = random.sample(image_files, 4)

plt.figure(figsize=(12,8))

for i, image_name in enumerate(random_images):

    # Image path
    image_path = os.path.join(image_folder, image_name)

    # Corresponding mask path
    mask_name = image_name.replace(".jpg", ".bmp")
    mask_path = os.path.join(mask_folder, mask_name)

    # Read image and mask
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    # Show image
    plt.subplot(4,2,2*i+1)
    plt.imshow(image)
    plt.title(image_name)
    plt.axis("off")

    # Show mask
    plt.subplot(4,2,2*i+2)
    plt.imshow(mask)
    plt.title(mask_name)
    plt.axis("off")

plt.tight_layout()
plt.show()