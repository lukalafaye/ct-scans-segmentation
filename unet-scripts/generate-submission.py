import os
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path

# Define paths
input_folder = Path("test-results-pp")  # Folder with postprocessed NIfTI files
output_csv_path = "predictions.csv"  # Output CSV file

# Get sorted list of all .nii.gz files
nii_files = sorted(input_folder.glob("*.nii.gz"), key=lambda x: int(x.stem.replace(".nii", "").split("_")[1]))  # Ensure correct order

print(f"Found {len(nii_files)} NIfTI files in {input_folder}")

# Initialize list to store flattened predictions
predictions_list = []
image_filenames = []  # Store filenames as "0.png", "1.png", etc.

# Process each prediction file
for nii_file in nii_files:
    # Extract numeric ID and format it as required (e.g., "0.png", "1.png")
    numeric_id = int(nii_file.stem.replace(".nii", "").split("_")[1])
    image_filename = f"{numeric_id}.png"

    # Load NIfTI file
    nii = nib.load(str(nii_file))
    segmentation = nii.get_fdata().astype(np.uint8)  # Convert to integer type

    # Ensure correct shape (256x256)
    if segmentation.shape[:2] != (256, 256):
        raise ValueError(f"Unexpected shape {segmentation.shape} in {nii_file}")

    # Flatten the 2D segmentation map
    predictions_list.append(segmentation[:, :, 0].flatten())
    image_filenames.append(image_filename)

# Convert list to 2D NumPy array (N images, 65536 pixels each)
predictions_array = np.array(predictions_list)  # Shape: (N, 65536)

# Transpose the matrix as required (to be 65536 x N)
predictions_transposed = predictions_array.T  # Shape: (65536, N)

# Define row indices as "Pixel 0", "Pixel 1", ..., "Pixel 65535"
row_index = [f"Pixel {i}" for i in range(65536)]

# Convert to Pandas DataFrame and set correct column names (0.png, 1.png, ...)
df = pd.DataFrame(predictions_transposed, columns=image_filenames)

# Rename index to match required format
df.index = row_index

# Save as CSV **without extra index column**
df.to_csv(output_csv_path, index=True)

print(f"✅ Predictions saved in {output_csv_path} with correct format.")
