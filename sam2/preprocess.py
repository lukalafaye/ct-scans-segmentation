import os

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ----------------------------
# Step 1. Configuration and Paths
# ----------------------------
# Define the base data folder (adjust as needed)
DATA_FOLDER = "../data"  # No trailing slash

# Define paths for original train images and the CSV file containing masks
TRAIN_IMAGES_DIR = os.path.join(DATA_FOLDER, "train-images")
Y_TRAIN_CSV = os.path.join(DATA_FOLDER, "y_train.csv")

# Define the output folder for the masks and for the new train CSV
MASKS_DIR = os.path.join(DATA_FOLDER, "masks")
os.makedirs(MASKS_DIR, exist_ok=True)

# ----------------------------
# Step 2. Load and Filter y_train.csv
# ----------------------------
# Load the CSV; note that the CSV is transposed so that each row corresponds to an image
df = pd.read_csv(Y_TRAIN_CSV, index_col=0).T
print("Original y_train.csv shape:", df.shape)

# Filter out rows that contain only zeros (no annotated mask)
mask = ~(df.values == 0).all(axis=-1)
df_filtered = df[mask]
print("Filtered train data shape:", df_filtered.shape)

# ----------------------------
# Step 3. Reshape and Save Masks as Images
# ----------------------------
# For each row in the filtered DataFrame, reshape the flattened mask into 256x256 and save as an image
for file_name in df_filtered.index:
    # Get the flattened mask as a numpy array (assumed type convertible to uint8)
    mask_flat = df_filtered.loc[file_name].values.astype(np.uint8)
    # Reshape into 256x256 (since 256*256 = 65536)
    mask_img = mask_flat.reshape(256, 256)

    # Replace all non-zero values with 0
    mask_img[mask_img != 0] = 1

    # Save the mask image into the MASKS_DIR with the same filename (e.g., "23.png")
    out_path = os.path.join(MASKS_DIR, file_name)
    cv2.imwrite(out_path, mask_img)

print("Saved masks to:", MASKS_DIR)

# Step 4. Create a New CSV File Mapping Filtered Images to Masks

# Use the filtered train_df index (which contains only images for which a mask was saved)
filtered_filenames = list(df_filtered.index)  # e.g. ['23.png', '45.png', ..., 'xxx.png']
print(filtered_filenames[:10])

# (Optionally, sort them numerically using a helper function)
import re
def numeric_key(filename):
    number_str = re.sub(r'\D', '', filename)
    return int(number_str) if number_str else float('inf')

filtered_filenames = sorted(filtered_filenames, key=numeric_key)

# Create a DataFrame with two columns: ImageId and MaskId.
# We assume that the mask files in the masks folder have the same filenames as in the filtered list.
train_csv_df = pd.DataFrame({
    "ImageId": filtered_filenames,
    "MaskId": filtered_filenames  # because masks are saved with the same names in data/masks
})

# Save this DataFrame as "train.csv" in your DATA_FOLDER.
TRAIN_CSV_PATH = os.path.join(DATA_FOLDER, "train.csv")
train_csv_df.to_csv(TRAIN_CSV_PATH, index=False)
print("Created new CSV mapping images to masks at:", TRAIN_CSV_PATH)

# ----------------------------
# Step 5. (Optional) Split the Data into Train and Test Sets
# ----------------------------
# If you wish to split your dataset into train and test sets for fine-tuning SAM2,
# you can use train_test_split. For example:
train_split, test_split = train_test_split(train_csv_df, test_size=0.2, random_state=42)
train_split.to_csv(os.path.join(DATA_FOLDER, "train_split.csv"), index=False)
test_split.to_csv(os.path.join(DATA_FOLDER, "test_split.csv"), index=False)
print("Saved train_split.csv and test_split.csv in", DATA_FOLDER)
