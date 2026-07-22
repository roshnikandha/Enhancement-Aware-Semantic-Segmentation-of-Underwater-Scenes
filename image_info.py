import cv2

# Load the image
image = cv2.imread("dataset/train_val/images/d_r_1_.jpg")

# Print information
print("Image Shape:", image.shape)
print("Height:", image.shape[0])
print("Width:", image.shape[1])
print("Channels:", image.shape[2])
print("Data Type:", image.dtype)