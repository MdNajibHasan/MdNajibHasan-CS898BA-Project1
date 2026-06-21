import cv2
from pathlib import Path
import numpy as np
import pandas as pd
#### Image path
IMAGE_PATH = "E:\Image Processing\Homework 1\Input\original_image.png"

###### Read image
image = cv2.imread(IMAGE_PATH)

### checking for image loading
if image is None:
    raise FileNotFoundError(f"Could not read image from: {IMAGE_PATH}")

print("Image loaded successfully!")
print("Image shape:", image.shape)
print("Height:", image.shape[0])
print("Width:", image.shape[1])
print("Channels:", image.shape[2])


##### Converting image from BGR to RGB as OpenCV reads images as BGR
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def calculate_channel_statistics(channel):
    """
    Calculate basic statistics for one image channel.
    """

    values = channel.flatten().astype(np.float64)

    min_value = np.min(values)
    max_value = np.max(values)
    average_value = np.mean(values)
    median_value = np.median(values)
    range_value = max_value - min_value
    std_value = np.std(values)
    variance_value = np.var(values)

    # Mode calculation for image pixel values from 0 to 255
    values_uint8 = values.astype(np.uint8)
    counts = np.bincount(values_uint8, minlength=256)
    mode_value = np.argmax(counts)

    # Skewness calculation
    if std_value == 0:
        skew_value = 0.0
    else:
        skew_value = np.mean(((values - average_value) / std_value) ** 3)

    return {
        "min": min_value,
        "max": max_value,
        "average": average_value,
        "median": median_value,
        "mode": mode_value,
        "skew": skew_value,
        "range": range_value,
        "standard_deviation": std_value,
        "variance": variance_value
    }


# Separate RGB channels
channels = {
    "Red": image_rgb[:, :, 0],
    "Green": image_rgb[:, :, 1],
    "Blue": image_rgb[:, :, 2]
}

# Calculate and print statistics for each channel
statistics_results = []

print("\nBasic Image Statistics for Each Channel")
print("=" * 50)

for channel_name, channel_data in channels.items():
    stats = calculate_channel_statistics(channel_data)

    print(f"\nChannel: {channel_name}")
    print("-" * 30)

    for stat_name, stat_value in stats.items():
        print(f"{stat_name}: {stat_value:.4f}")

    row = {"channel": channel_name}
    row.update(stats)
    statistics_results.append(row)