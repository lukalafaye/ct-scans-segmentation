import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Directories
TEST_DIR = "test-images"
PRED_DIR = "predictions"
SAVE_DIR = "pred_with_originals"

# Create the save directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

# Get a sorted list of test image filenames (assuming PNG files)
test_files = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith(".png")])

for file_name in test_files:
    test_path = os.path.join(TEST_DIR, file_name)
    pred_path = os.path.join(PRED_DIR, file_name)
    
    # Check if the corresponding prediction file exists
    if not os.path.exists(pred_path):
        print(f"Predicted file not found for {file_name}, skipping...")
        continue

    # Load the test image in color and convert from BGR to RGB
    test_img = cv2.imread(test_path, cv2.IMREAD_COLOR)
    if test_img is None:
        print(f"Failed to load test image: {test_path}")
        continue
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    
    # Load the predicted mask in grayscale
    pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    if pred_img is None:
        print(f"Failed to load predicted mask: {pred_path}")
        continue

    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the test image
    axes[0].imshow(test_img)
    axes[0].set_title(f"Test Image: {file_name}")
    axes[0].axis("off")

    # Plot the predicted mask with a colormap
    axes[1].imshow(pred_img, cmap="jet")
    axes[1].set_title(f"Predicted Mask: {file_name}")
    axes[1].axis("off")

    plt.tight_layout()
    
    # Save the figure to the SAVE_DIR using the original filename
    save_path = os.path.join(SAVE_DIR, file_name)
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved combined image to {save_path}")
