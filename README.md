# Enhancement-Aware Semantic Segmentation of Underwater Scenes

## Project Overview

This project focuses on improving semantic segmentation of underwater images by applying image enhancement before segmentation. Underwater images often suffer from poor visibility, low contrast, color distortion, and haze, which reduce segmentation accuracy.

Our pipeline first enhances underwater images using CLAHE (Contrast Limited Adaptive Histogram Equalization) and then performs semantic segmentation using DeepLabV3 with a ResNet-50 backbone.

---

## Features

- Underwater image enhancement using CLAHE
- Semantic segmentation using DeepLabV3
- Supports 8 underwater object classes
- Model evaluation using IoU and Pixel Accuracy
- Streamlit web application for prediction
- Visualization of segmentation masks

---

## Dataset

Dataset Used: SUIM (Segmented Underwater Image Dataset)

Number of Training Images: 1525

Number of Test Images: 110

Classes:

- Background Water
- Human Divers
- Plants / Sea Grass
- Wrecks / Ruins
- Robots / Instruments
- Reefs & Invertebrates
- Fish & Vertebrates
- Sand / Sea Floor

---

## Project Pipeline

Input Image

↓

CLAHE Enhancement

↓

DeepLabV3 Semantic Segmentation

↓

Predicted Segmentation Mask

↓

Evaluation (IoU & Pixel Accuracy)

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- OpenCV
- NumPy
- Matplotlib
- Streamlit

---

## Folder Structure

```
app.py
clahe_enhancement.py
comparison.py

models/
enhancement/
results/
checkpoints/

requirements.txt
README.md
```

---

## Installation

```bash
git clone <repository-link>

cd enhancement-aware-semantic-segmentation-of-underwater-scenes

pip install -r requirements.txt

streamlit run app.py
```

---

## Results

The project successfully performs semantic segmentation on underwater images after enhancement.

Generated Outputs:

- Original Image
- CLAHE Enhanced Image
- Predicted Segmentation Mask

Evaluation Metrics:

- Pixel Accuracy
- Mean IoU
- IoU Per Class

---

## Future Improvements

- Train for more epochs
- Add multiple enhancement techniques
- Improve segmentation accuracy
- Deploy on cloud
- Real-time underwater video segmentation

---

## Team

IGDTUW

Computer Vision and Deep Learning Project
