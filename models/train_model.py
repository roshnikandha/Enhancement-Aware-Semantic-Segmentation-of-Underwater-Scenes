import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

from dataset import SUIMDataset

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
dataset = SUIMDataset(
    "dataset/train_val/images",
    "dataset/train_val/masks"
)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

# Model
model = deeplabv3_resnet50(weights="DEFAULT")
model.classifier = DeepLabHead(2048, 8)
model = model.to(device)

# Loss
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)

print("Everything Loaded Successfully!")
print("Starting one training step...")

model.train()

images, masks = next(iter(loader))

images = images.to(device)
masks = masks.to(device)

optimizer.zero_grad()

outputs = model(images)["out"]

loss = criterion(outputs, masks)

loss.backward()

optimizer.step()

print("Training Step Complete!")
print("Loss:", loss.item())