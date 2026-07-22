import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

from dataset import SUIMDataset

# ---------------------------------
# Device
# ---------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---------------------------------
# Dataset
# ---------------------------------
dataset = SUIMDataset(
    "dataset/train_val/images",
    "dataset/train_val/masks"
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False
)

# ---------------------------------
# Model
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

model = model.to(device)
model.eval()

print("Model Loaded Successfully!")

# ---------------------------------
# Variables
# ---------------------------------
correct_pixels = 0
total_pixels = 0

intersection = np.zeros(8)
union = np.zeros(8)

print("Starting Evaluation...\n")

# ---------------------------------
# Evaluate ONLY first 20 images
# ---------------------------------
with torch.no_grad():

    for idx, (images, masks) in enumerate(loader):

        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)["out"]
        preds = torch.argmax(outputs, dim=1)

        correct_pixels += (preds == masks).sum().item()
        total_pixels += masks.numel()

        preds_np = preds.cpu().numpy()
        masks_np = masks.cpu().numpy()

        for cls in range(8):

            pred_cls = (preds_np == cls)
            mask_cls = (masks_np == cls)

            intersection[cls] += np.logical_and(pred_cls, mask_cls).sum()
            union[cls] += np.logical_or(pred_cls, mask_cls).sum()

        print(f"Processed Image {idx + 1}/20")

        if idx >= 19:
            break

# ---------------------------------
# Metrics
# ---------------------------------
pixel_accuracy = correct_pixels / total_pixels

iou = intersection / (union + 1e-6)

mean_iou = np.mean(iou)

print("\n==============================")
print(f"Pixel Accuracy : {pixel_accuracy:.4f}")
print(f"Mean IoU       : {mean_iou:.4f}")
print("==============================")

print("\nIoU Per Class")

for i in range(8):
    print(f"Class {i}: {iou[i]:.4f}")