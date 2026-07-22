import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

# ---------------------------------
# Device
# ---------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---------------------------------
# Create results folder
# ---------------------------------
os.makedirs("results", exist_ok=True)

# ---------------------------------
# Load Model
# ---------------------------------
model = deeplabv3_resnet50(
    weights=None,
    weights_backbone=None,
    aux_loss=True
)

model.classifier = DeepLabHead(2048, 8)
model.aux_classifier[4] = torch.nn.Conv2d(256, 8, kernel_size=1)

model.load_state_dict(
    torch.load("checkpoints/suim_model.pth", map_location=device)
)

model.to(device)
model.eval()

print("Model Loaded Successfully!")

# ---------------------------------
# Image Path
# ---------------------------------
image_path = "dataset/TEST/images/d_r_47_.jpg"

image = cv2.imread(image_path)

if image is None:
    print("Image not found!")
    exit()

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

original = image.copy()

image = cv2.resize(image, (512, 512))

image_tensor = torch.tensor(image).permute(2,0,1).float()/255.0
image_tensor = image_tensor.unsqueeze(0).to(device)

# ---------------------------------
# Prediction
# ---------------------------------
with torch.no_grad():
    output = model(image_tensor)["out"]

prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()

# ---------------------------------
# Save Prediction
# ---------------------------------
plt.imsave("results/predicted_mask.png", prediction)

# ---------------------------------
# Save Original
# ---------------------------------
plt.imsave("results/original_image.png", original)

# ---------------------------------
# Save Comparison
# ---------------------------------
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.imshow(original)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(prediction)
plt.title("Predicted Mask")
plt.axis("off")

plt.tight_layout()

plt.savefig("results/comparison.png", dpi=300)

plt.show()

print("\nSaved Successfully!")
print("results/original_image.png")
print("results/predicted_mask.png")
print("results/comparison.png")