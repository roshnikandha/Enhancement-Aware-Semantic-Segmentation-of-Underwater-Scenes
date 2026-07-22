import os
import torch
import torch.nn as nn
import torch.optim as optim
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
    batch_size=2,
    shuffle=True
)

# ---------------------------------
# Model
# ---------------------------------
model = deeplabv3_resnet50(
    weights="DEFAULT",
    aux_loss=True
)

# Main classifier
model.classifier = DeepLabHead(2048, 8)

# Auxiliary classifier
model.aux_classifier[4] = nn.Conv2d(
    256,
    8,
    kernel_size=1
)

model = model.to(device)

# ---------------------------------
# Loss
# ---------------------------------
criterion = nn.CrossEntropyLoss()

# ---------------------------------
# Optimizer
# ---------------------------------
optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001
)

# ---------------------------------
# Checkpoint Folder
# ---------------------------------
os.makedirs("checkpoints", exist_ok=True)

epochs = 1

print("Starting Training...\n")

for epoch in range(epochs):

    model.train()

    running_loss = 0

    for batch_idx, (images, masks) in enumerate(loader):

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)["out"]

        loss = criterion(outputs, masks)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        print(f"Batch {batch_idx+1} Loss: {loss.item():.4f}")

        # Stop after 20 batches
        if batch_idx == 19:
            break

    print(f"\nEpoch {epoch+1} Average Loss: {running_loss/20:.4f}")

torch.save(
    model.state_dict(),
    "checkpoints/suim_model.pth"
)

print("\nTraining Complete!")
print("Model saved successfully.")