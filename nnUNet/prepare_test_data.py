import os
import cv2
import nibabel as nib
import numpy as np
from pathlib import Path

# Define dataset ID and name
dataset_id = "001"
dataset_name = "CTSCAN"
dataset_folder = f"nnUNet_raw/Dataset{dataset_id}_{dataset_name}"

# Create test images directory
imagesTs_dir = os.path.join(dataset_folder, "imagesTs")
os.makedirs(imagesTs_dir, exist_ok=True)

print(f"Created test dataset folder at {imagesTs_dir}")

# Path to the raw test images folder
test_image_dir = Path("./test-images")

# Get all image files (assuming filenames are like "0.png", "1.png", etc.)
test_image_files = sorted(test_image_dir.glob("*.png"), key=lambda x: int(x.stem))

print(f"Found {len(test_image_files)} test images.")

# Convert PNG images to NIfTI format for nnU-Net
for idx, img_path in enumerate(test_image_files):
    case_id = f"case_{idx:04d}"  # create a new case id
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
    img_resized = cv2.resize(img, (256, 256))  # Resize to 256x256 if needed
    img_nii = nib.Nifti1Image(img_resized[..., np.newaxis], affine=np.eye(4))  # Convert to NIfTI
    nib.save(img_nii, os.path.join(imagesTs_dir, f"{case_id}_0000.nii.gz"))  # Save as NIfTI

print(f"Converted {len(test_image_files)} test images to NIfTI format.")

# Ensure environment variables are set
os.environ["nnUNet_raw"] = "nnUNet_raw"
os.environ["nnUNet_preprocessed"] = "nnUNet_preprocessed"
os.environ["nnUNet_results"] = "nnUNet_results"

# Create directories if they do not exist
os.makedirs(os.environ["nnUNet_raw"], exist_ok=True)
os.makedirs(os.environ["nnUNet_preprocessed"], exist_ok=True)
os.makedirs(os.environ["nnUNet_results"], exist_ok=True)

print("✅ Test images converted successfully and environment variables set.")
