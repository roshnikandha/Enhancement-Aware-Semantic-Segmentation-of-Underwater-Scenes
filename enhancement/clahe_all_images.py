import os
import cv2

input_folder = "dataset/train_val/images"
output_folder = "enhancement/clahe_dataset"

os.makedirs(output_folder, exist_ok=True)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

image_files = os.listdir(input_folder)

print(f"Found {len(image_files)} images.")

for image_name in image_files:

    image_path = os.path.join(input_folder, image_name)

    image = cv2.imread(image_path)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    l = clahe.apply(l)

    enhanced = cv2.merge((l, a, b))

    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    save_path = os.path.join(output_folder, image_name)

    cv2.imwrite(save_path, enhanced)

print("🎉 All images enhanced successfully!")