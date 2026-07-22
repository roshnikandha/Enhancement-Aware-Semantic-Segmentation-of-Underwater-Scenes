import cv2
import matplotlib.pyplot as plt

# Read image
image = cv2.imread("dataset/train_val/images/d_r_1_.jpg")

# Convert BGR to RGB for display
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to LAB color space
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# Split LAB channels
l, a, b = cv2.split(lab)

# Apply CLAHE to brightness channel
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
l_clahe = clahe.apply(l)

# Merge channels
lab_clahe = cv2.merge((l_clahe, a, b))

# Convert back to BGR
enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

# Convert to RGB
enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

# Save enhanced image
cv2.imwrite("enhancement/outputs/clahe_d_r_1_.jpg", enhanced)

print("Enhanced image saved successfully!")

# Display
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.imshow(image_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(enhanced_rgb)
plt.title("CLAHE Enhanced")
plt.axis("off")

plt.tight_layout()
plt.show()