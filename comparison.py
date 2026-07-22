import os
import cv2
import torch
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
# Load Image
# ---------------------------------
image = cv2.imread("dataset/train_val/images/d_r_1_.jpg")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ---------------------------------
# CLAHE Enhancement
# ---------------------------------
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

l, a, b = cv2.split(lab)

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8,8)
)

l = clahe.apply(l)

lab = cv2.merge((l, a, b))

enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

# ---------------------------------
# Load Model
# ---------------------------------
model = deeplabv3_resnet50(
    weights=None,
    weights_backbone=None,
    aux_loss=True
)

model.classifier = DeepLabHead(2048,8)
model.aux_classifier[4] = torch.nn.Conv2d(256,8,1)

model.load_state_dict(
    torch.load(
        "checkpoints/suim_model.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

# ---------------------------------
# Prediction
# ---------------------------------
img = cv2.resize(enhanced_rgb,(512,512))

img = torch.tensor(img).permute(2,0,1).float()/255.0
img = img.unsqueeze(0).to(device)

with torch.no_grad():
    output = model(img)["out"]

prediction = torch.argmax(output,1).squeeze().cpu().numpy()

# ---------------------------------
# Save Figure
# ---------------------------------
plt.figure(figsize=(18,6))

plt.subplot(1,3,1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(enhanced_rgb)
plt.title("CLAHE Enhanced")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(prediction)
plt.title("Segmentation Result")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    "results/original_clahe_prediction.png",
    dpi=300
)

plt.show()

print("Saved to results/original_clahe_prediction.png")