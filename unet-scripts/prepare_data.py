"""
BEFORE


pip3 install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -q nnunetv2
pip install -q triton

# downloading the images

wget -q https://challengedata.ens.fr/media/public/train-images.zip
wget -q https://challengedata.ens.fr/media/public/test-images.zip
wget -q https://challengedata.ens.fr/media/public/label_Hnl61pT.csv -O y_train.csv
wget -q https://challengedata.ens.fr/media/public/annotated_labels.json

# Unzip images

unzip -q -n train-images.zip
unzip -q -n test-images.zip

AFER
nnUNetv2_plan_and_preprocess -d 001 -pl nnUNetPlannerResEncM --verify_dataset_integrity
nnUNet_compile=False nnUNetv2_train Dataset001_CTSCAN 2d 0 -p nnUNetResEncUNetMPlans --npz
"""

from skimage.segmentation import watershed, felzenszwalb
from skimage.filters import sobel
import pandas as pd
from pathlib import Path
import cv2
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.filters import rank
from scipy import ndimage as ndi
from skimage.morphology import disk
import sklearn.metrics
import json
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

import datetime
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
import triton
import os 

# Check if GPU is available
print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))

# Ensure TensorFlow uses GPU
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

# Load the train labels
# Note the transpose!
labels_train = pd.read_csv("y_train.csv", index_col=0).T

# load annotated_labels.json

with open("annotated_labels.json", "r") as f:
    annotated_labels = json.load(f)

# Here is a function to load the data
def load_dataset(dataset_dir):
    dataset_list = []
    # Note: It's very important to load the images in the correct numerical order!
    for image_file in list(sorted(Path(dataset_dir).glob("*.png"), key=lambda filename: int(filename.name.rstrip(".png")))):
        dataset_list.append(cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE))
    return np.stack(dataset_list, axis=0)

# Load the train and test sets
# If you've put the shortcut directly in your drive, this should work out of the box
# Else, edit the path
data_dir = Path("./")
data_train = load_dataset(data_dir / "train-images")
data_test = load_dataset(data_dir / "test-images")

# Each image in annotated_labels contains 27 clusters,
# only some of these were actually annotated pixel by pixel in labels_train
# hence most cluster ids are not represented on most images and replaced by 0s..

df_annotated_labels = pd.DataFrame(annotated_labels).fillna(-1)
df_annotated_labels.index = [f"{i}.png" for i in range(len(annotated_labels))]  # Add .png extension

# The train data is a numpy array of 1000 images of 512*512
print(f"X_train shape: {data_train.shape}")
# The train label is a dataframe of 1000 rows with 262144 (=512x512) columns
print(f"Y_train shape: {labels_train.shape}")

# Boolean mask for rows that contain only zeros
zero_rows = labels_train.eq(0).all(axis=1)

# Count rows that contain only zeros (unannotated images)
unannotated_labels_train = zero_rows.sum()

print(f"Number of unannotated images in labels_train: {unannotated_labels_train}")

# Boolean mask for rows that contain at least one nonzero pixel
annotated_rows = labels_train.ne(0).any(axis=1)

# Filter only the 800 images with labeled structures
labels_train_annotated = labels_train[annotated_rows]

# Print the number of remaining images
print(f"Number of annotated images in labels_train: {labels_train_annotated.shape[0]}")

# Count the number of images with empty lists (unannotated)
unannotated_json = sum(len(labels) == 0 for labels in annotated_labels)

print(f"Number of unannotated images in annotated_labels.json: {unannotated_json}")

# Count the number of images with empty lists (unannotated)
annotated_json = sum(len(labels) > 0 for labels in annotated_labels)

print(f"Number of annotated images in annotated_labels.json: {annotated_json}")

# Get list of all image names in labels_train that are fully zero
unannotated_labels_train_set = set(labels_train.index[zero_rows])

# Get list of all unannotated image names from JSON
unannotated_json_set = {f"{i}.png" for i, labels in enumerate(annotated_labels) if len(labels) == 0}

# Find the extra unannotated images in labels_train (but not in JSON)
extra_unannotated_in_csv = unannotated_labels_train_set - unannotated_json_set

print(f"Number of extra unannotated images in labels_train: {len(extra_unannotated_in_csv)}")
print(extra_unannotated_in_csv)

# Labels are 0 for background and 1 to 54 for all organs
print(sorted(labels_train.stack().unique()))

# Example

image_id = 39
labels_for_image = annotated_labels[image_id]

print(f"{len(labels_for_image)} annotated labels for image {image_id}.png: {labels_for_image}")

print(f"Number of annotated images in labels_train: {labels_train_annotated.shape[0]}")

# Get the indices (or filenames) of annotated images.
# (Assuming that the index of labels_df is the image filename, e.g. "0.png", "1.png", etc.)
annotated_filenames = labels_train_annotated.index.tolist()
print("Annotated filenames (first 10):", annotated_filenames[:10])

# Define dataset ID and name (must follow the pattern: DatasetXXX_Name)
dataset_id = "001"
dataset_name = "CTSCAN"
dataset_folder = f"nnUNet_raw/Dataset{dataset_id}_{dataset_name}"

# Create the necessary subdirectories: imagesTr and labelsTr
imagesTr_dir = os.path.join(dataset_folder, "imagesTr")
labelsTr_dir = os.path.join(dataset_folder, "labelsTr")
os.makedirs(imagesTr_dir, exist_ok=True)
os.makedirs(labelsTr_dir, exist_ok=True)

print(f"Created dataset folder structure at {dataset_folder}")

import cv2
import nibabel as nib
from pathlib import Path

# Path to the raw images folder
image_dir = Path("./train-images")
# Get all image files (assuming filenames are like "0.png", "1.png", etc.)
all_image_files = sorted(image_dir.glob("*.png"), key=lambda x: int(x.stem))

# Filter the image files to only those that are annotated.
# We assume that the filename (e.g. "42.png") is in our annotated_filenames list.
annotated_image_files = [img for img in all_image_files if img.name in annotated_filenames]
print(f"Found {len(annotated_image_files)} annotated images.")

# Loop over the filtered annotated image files and convert them to NIfTI
for idx, img_path in enumerate(annotated_image_files):
    case_id = f"case_{idx:04d}"  # create new case id based on order in filtered list
    # Read image in grayscale
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    # Resize to 256x256 (if necessary)
    img_resized = cv2.resize(img, (256, 256))
    # Expand dimensions to add a channel dimension (256x256x1)
    img_nii = nib.Nifti1Image(img_resized[..., np.newaxis], affine=np.eye(4))
    # Save image NIfTI using the nnU-Net naming convention (channel identifier _0000)
    nib.save(img_nii, os.path.join(imagesTr_dir, f"{case_id}_0000.nii.gz"))

print(f"Converted {len(annotated_image_files)} annotated images to NIfTI format.")

import numpy as np
import nibabel as nib

# Convert the filtered labels DataFrame into a NumPy array and reshape to (N, 256, 256)
labels_np = labels_train_annotated.values.reshape((-1, 256, 256)).astype(np.uint8)
num_annotated = labels_np.shape[0]
print(f"Number of annotated labels: {num_annotated}")

# Loop over the annotated labels and convert them to NIfTI
for idx in range(num_annotated):
    case_id = f"case_{idx:04d}"  # should match the case id used for images
    label_nii = nib.Nifti1Image(labels_np[idx][..., np.newaxis], affine=np.eye(4))
    nib.save(label_nii, os.path.join(labelsTr_dir, f"{case_id}.nii.gz"))

print(f"Converted {num_annotated} annotated labels to NIfTI format.")

import os

os.environ["nnUNet_raw"] = "nnUNet_raw"
os.environ["nnUNet_preprocessed"] = "nnUNet_preprocessed"
os.environ["nnUNet_results"] = "nnUNet_results"

# Create directories if they do not exist
os.makedirs(os.environ["nnUNet_raw"], exist_ok=True)
os.makedirs(os.environ["nnUNet_preprocessed"], exist_ok=True)
os.makedirs(os.environ["nnUNet_results"], exist_ok=True)

print("Environment variables set correctly.")

import json
import os

dataset_json_path = os.path.join(dataset_folder, "dataset.json")

# Define modality (for example, CT)
channel_names = {"0": "CT"}

# Define labels (0: background, 1-54: organs)
labels_dict = {"background": 0}
for i in range(1, 55):
    labels_dict[f"organ_{i}"] = i

# Count training images (should equal the number of annotated images)
num_training = len([f for f in os.listdir(imagesTr_dir) if f.endswith(".nii.gz")])

# Build the dataset.json dictionary
dataset_json = {
    "channel_names": channel_names,
    "labels": labels_dict,
    "numTraining": num_training,
    "file_ending": ".nii.gz",
    "overwrite_image_reader_writer": "SimpleITKIO"
}

# For each training case, add an entry (nnU-Net uses relative paths)
training_entries = []
for idx in range(num_training):
    case_id = f"case_{idx:04d}"
    entry = {
        "image": f"./imagesTr/{case_id}_0000.nii.gz",
        "label": f"./labelsTr/{case_id}.nii.gz"
    }
    training_entries.append(entry)
dataset_json["training"] = training_entries
dataset_json["test"] = []  # (if you have test cases, populate here)

# Save the dataset.json file
with open(dataset_json_path, "w") as f:
    json.dump(dataset_json, f, indent=4)

print(f"✅ dataset.json created at: {dataset_json_path}")
