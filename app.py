import os
import gdown
import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image

from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

# -------------------------------------------------
# Download model automatically if not present
# -------------------------------------------------

MODEL_PATH = "checkpoints/suim_model.pth"

if not os.path.exists(MODEL_PATH):
    os.makedirs("checkpoints", exist_ok=True)

    gdown.download(
        "https://drive.google.com/uc?id=1kidcwb_Tn-3sZndNsx9B6aQbeL0soZUA",
        MODEL_PATH,
        quiet=False,
    )


# -------------------------------------------------
# Load Model
# -------------------------------------------------

@st.cache_resource
def load_model():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            MODEL_PATH,
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    return model, device


model, device = load_model()

# -------------------------------------------------
# Sidebar
# -------------------------------------------------

st.sidebar.title("Model Information")

st.sidebar.markdown("""
**Model:** DeepLabV3 + ResNet50

**Dataset:** SUIM

**Classes:** 8

**Framework:** PyTorch

**Deployment:** Streamlit Cloud
""")

# -------------------------------------------------
# Color Mask
# -------------------------------------------------

def colorize_mask(mask):

    colors = np.array([
        [0, 0, 0],        # Background
        [255, 0, 0],      # Red
        [0, 255, 0],      # Green
        [0, 0, 255],      # Blue
        [255, 255, 0],    # Yellow
        [255, 0, 255],    # Magenta
        [0, 255, 255],    # Cyan
        [255, 255, 255],  # White
    ])

    return colors[mask].astype(np.uint8)


# -------------------------------------------------
# Prediction
# -------------------------------------------------

def predict(image):

    image = np.array(image.convert("RGB"))

    original = image.copy()

    image = cv2.resize(image, (512, 512))

    image_tensor = torch.tensor(image)

    image_tensor = image_tensor.permute(2, 0, 1)

    image_tensor = image_tensor.float() / 255.0

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    with torch.no_grad():

        output = model(image_tensor)["out"]

    prediction = torch.argmax(output, dim=1)

    prediction = prediction.squeeze().cpu().numpy()

    colored_mask = colorize_mask(prediction)

    return original, colored_mask


# -------------------------------------------------
# UI
# -------------------------------------------------

st.title("🌊 Enhancement-Aware Semantic Segmentation of Underwater Scenes")

st.write(
    "DeepLabV3 + ResNet50 | SUIM Dataset | Semantic Segmentation"
)

st.markdown("---")

st.subheader("Class Color Legend")

st.markdown("""
⬛ **Black** – Background

🟥 **Red** – Human Diver

🟩 **Green** – Aquatic Plants

🟦 **Blue** – Wrecks / Robots

🟨 **Yellow** – Reefs

🟪 **Magenta** – Water

🩵 **Cyan** – Fish

⬜ **White** – Sea Floor
""")

st.markdown("---")

uploaded = st.file_uploader(
    "Upload an underwater image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:

    image = Image.open(uploaded)

    with st.spinner("Running semantic segmentation..."):

        original, mask = predict(image)

    st.success("Segmentation Complete!")

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

st.markdown("---")

st.caption(
    "Computer Vision Project • Enhancement-Aware Semantic Segmentation of Underwater Scenes"
)