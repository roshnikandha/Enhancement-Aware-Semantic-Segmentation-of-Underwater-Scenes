import cv2
import matplotlib.pyplot as plt

# Read the original image
image = cv2.imread("dataset/train_val/images/d_r_1_.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Read the mask
mask = cv2.imread("dataset/train_val/masks/d_r_1_.bmp")
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

# Create a bigger figure
plt.figure(figsize=(10,5))

# Show original image
plt.subplot(1,2,1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

# Show segmentation mask
plt.subplot(1,2,2)
plt.imshow(mask)
plt.title("Segmentation Mask")
plt.axis("off")

plt.show()