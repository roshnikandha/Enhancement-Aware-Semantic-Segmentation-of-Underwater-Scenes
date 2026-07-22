import torch
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained model
model = deeplabv3_resnet50(weights="DEFAULT")

# Change the classifier to output 8 classes
model.classifier = DeepLabHead(2048, 8)

model = model.to(device)

model.eval()

print("Device:", device)
print("Model Modified Successfully!")

dummy = torch.randn(1,3,480,640).to(device)

with torch.no_grad():
    output = model(dummy)

print("Output Shape:", output["out"].shape)