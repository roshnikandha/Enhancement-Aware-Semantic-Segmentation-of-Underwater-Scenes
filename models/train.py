from dataset import SUIMDataset
from torch.utils.data import DataLoader
import torch

dataset = SUIMDataset(
    "dataset/train_val/images",
    "dataset/train_val/masks"
)

loader = DataLoader(dataset, batch_size=2, shuffle=True)

images, masks = next(iter(loader))

print("Images:", images.shape)
print("Masks:", masks.shape)

print("Unique Classes:", torch.unique(masks))