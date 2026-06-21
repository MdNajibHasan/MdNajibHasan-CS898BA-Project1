import cv2
from pathlib import Path

#### Image path
IMAGE_PATH = "E:\Image Processing\Homework 1\Input\original_image.png"

###### Read image
image = cv2.imread(IMAGE_PATH)

# Check if image was loaded correctly
if image is None:
    raise FileNotFoundError(f"Could not read image from: {IMAGE_PATH}")

print("Image loaded successfully!")
print("Image shape:", image.shape)
print("Height:", image.shape[0])
print("Width:", image.shape[1])
print("Channels:", image.shape[2])