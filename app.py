import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image

from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


# -----------------------
# Load Model
# -----------------------

@st.cache_resource
def load_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = deeplabv3_resnet50(
        weights=None,
        weights_backbone=None,
        aux_loss=True
    )

    model.classifier = DeepLabHead(2048, 8)

    model.aux_classifier[4] = torch.nn.Conv2d(
        256,
        8,
        kernel_size=1
    )

    model.load_state_dict(
        torch.load(
            "checkpoints/suim_model.pth",
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    return model, device


model, device = load_model()


# -----------------------
# Color Mask
# -----------------------

def colorize_mask(mask):

    colors = np.array([
        [0, 0, 0],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
        [255, 255, 255]
    ])

    colored = colors[mask]

    return colored.astype(np.uint8)



# -----------------------
# Prediction
# -----------------------

def predict(image):

    # FIX: convert RGBA/PNG images to RGB
    image = np.array(image.convert("RGB"))

    original = image.copy()


    image = cv2.resize(
        image,
        (512, 512)
    )


    image_tensor = torch.tensor(image)

    image_tensor = image_tensor.permute(
        2, 0, 1
    )

    image_tensor = image_tensor.float() / 255.0

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)


    with torch.no_grad():

        output = model(image_tensor)["out"]


    prediction = torch.argmax(
        output,
        dim=1
    )


    prediction = prediction.squeeze().cpu().numpy()


    colored_mask = colorize_mask(prediction)


    return original, colored_mask



# -----------------------
# Streamlit UI
# -----------------------

st.title(
    "Enhancement-Aware Semantic Segmentation of Underwater Scenes"
)


uploaded = st.file_uploader(
    "Upload underwater image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


if uploaded:

    image = Image.open(uploaded)


    original, mask = predict(image)


    col1, col2 = st.columns(2)


    with col1:
        st.subheader("Original Image")
        st.image(
            original,
            use_container_width=True
        )


    with col2:
        st.subheader("Segmentation Mask")
        st.image(
            mask,
            use_container_width=True
        )