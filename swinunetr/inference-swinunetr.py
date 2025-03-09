#!/usr/bin/env python3
import os
import re
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from monai.networks.nets import SwinUNETR  # MONAI's implementation
import albumentations as A

# ----------------------------
# Step 1: Configuration and Data Paths
# ----------------------------
DATA_FOLDER = "../data"  # no trailing slash
TRAIN_IMAGES_DIR = os.path.join(DATA_FOLDER, "train-images")
TEST_IMAGES_DIR = os.path.join(DATA_FOLDER, "test-images")
Y_TRAIN_CSV = os.path.join(DATA_FOLDER, "y_train.csv")

# Output folders for visualizations
OUTPUT_TRAIN_DIR = os.path.join(DATA_FOLDER, "train-inference-swinunetr")
OUTPUT_TEST_DIR = os.path.join(DATA_FOLDER, "test-swinunetr")
os.makedirs(OUTPUT_TRAIN_DIR, exist_ok=True)
os.makedirs(OUTPUT_TEST_DIR, exist_ok=True)

IMG_SIZE = 256      # images are 256x256
NUM_CLASSES = 55    # class 0 = background, 1..54 = organs

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Step 2: Load and Filter CSV Masks (for train images)
# ----------------------------
df = pd.read_csv(Y_TRAIN_CSV, index_col=0).T
print("Original y_train.csv shape:", df.shape)
# Filter out rows that are all zeros (i.e. no annotations)
mask = ~(df.values == 0).all(axis=1)
df_filtered = df[mask]
print("Filtered train data shape:", df_filtered.shape)

# Build a dictionary: filename -> ground truth mask (reshaped to 256x256)
gt_masks = {}
for fname in df_filtered.index:
    mask_flat = df_filtered.loc[fname].values.astype(np.uint8)
    if mask_flat.shape[0] != IMG_SIZE * IMG_SIZE:
        raise ValueError(f"Mask for {fname} has {mask_flat.shape[0]} elements; expected {IMG_SIZE*IMG_SIZE}")
    gt_masks[fname] = mask_flat.reshape(IMG_SIZE, IMG_SIZE)

# ----------------------------
# Step 3: Define Inference Helper Functions
# ----------------------------

def preprocess_image(image_path):
    """
    Reads a grayscale image, resizes to IMG_SIZE x IMG_SIZE, scales to [0,1],
    and returns a torch tensor of shape (1,1,IMG_SIZE,IMG_SIZE).
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.astype(np.float32) / 255.0
    return torch.tensor(image).unsqueeze(0).unsqueeze(0)

def infer_mask(model, image_path):
    """
    Given an image path, runs inference on the image using the model and returns
    the predicted segmentation mask as a numpy array of shape (IMG_SIZE, IMG_SIZE).
    """
    input_tensor = preprocess_image(image_path).to(device)
    with torch.no_grad():
        logits = model(input_tensor)  # (1, NUM_CLASSES, IMG_SIZE, IMG_SIZE)
        pred = torch.argmax(logits, dim=1).squeeze(0)  # (IMG_SIZE, IMG_SIZE)
    return pred.cpu().numpy()

def apply_colormap(mask):
    """
    Converts a single-channel mask (with values 0..NUM_CLASSES-1) to a colored image.
    Scales the mask values appropriately and applies a JET colormap.
    """
    mask_uint8 = np.uint8(mask)
    scale = 255 // (NUM_CLASSES - 1) if NUM_CLASSES > 1 else 255
    mask_scaled = mask_uint8 * scale
    colored = cv2.applyColorMap(mask_scaled, cv2.COLORMAP_JET)
    return colored

def overlay_mask(image, colored_mask, alpha=0.5):
    """
    Overlays a colored mask onto the original grayscale image.
    """
    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(image_bgr, 1 - alpha, colored_mask, alpha, 0)

# ----------------------------
# Step 4: Load the Model from Checkpoint
# ----------------------------
def load_model(checkpoint_path):
    """
    Loads the MONAI SwinUNETR model with the given checkpoint.
    Adjusts the state dict keys by removing a 'model.' prefix if necessary.
    """
    model = SwinUNETR(
        img_size=(IMG_SIZE, IMG_SIZE),
        in_channels=1,
        out_channels=NUM_CLASSES,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        feature_size=48,
        norm_name="instance",
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        normalize=True,
        spatial_dims=2
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Assume checkpoint contains a state dict under key "state_dict"
    state_dict = checkpoint.get("state_dict", checkpoint)
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("model.", "")  # remove prefix if present
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model

# ----------------------------
# Step 5: Inference on Train and Test Images
# ----------------------------
def inference_on_train(model):
    """
    Runs inference on all train images (with GT available) and saves a side-by-side visualization:
    original image, ground truth mask (colored), and predicted mask overlay.
    """
    for fname, gt_mask in gt_masks.items():
        image_path = os.path.join(TRAIN_IMAGES_DIR, fname)
        try:
            original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if original is None:
                print(f"Skipping {fname}: image not found.")
                continue
            original = cv2.resize(original, (IMG_SIZE, IMG_SIZE))
            pred_mask = infer_mask(model, image_path)
            gt_colored = apply_colormap(gt_mask)
            pred_colored = apply_colormap(pred_mask)
            overlayed_pred = overlay_mask(original, pred_colored, alpha=0.5)
            
            # Create a side-by-side figure
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(original, cmap="gray")
            axs[0].set_title("Original Image")
            axs[0].axis("off")
            axs[1].imshow(gt_colored)
            axs[1].set_title("Ground Truth Mask")
            axs[1].axis("off")
            axs[2].imshow(overlayed_pred)
            axs[2].set_title("Prediction Overlay")
            axs[2].axis("off")
            plt.tight_layout()
            save_path = os.path.join(OUTPUT_TRAIN_DIR, fname)
            plt.savefig(save_path)
            plt.close(fig)
            print(f"Saved train inference visualization for {fname}")
        except Exception as e:
            print(f"Error processing {fname}: {e}")

def inference_on_test(model):
    """
    Runs inference on test images (from TEST_IMAGES_DIR), saves an overlayed prediction,
    and creates a submission CSV file with flattened predictions.
    """
    submission_dict = {}
    test_files = [f for f in os.listdir(TEST_IMAGES_DIR) if re.search(r'\.(png|jpg|jpeg)$', f, re.IGNORECASE)]
    for fname in test_files:
        image_path = os.path.join(TEST_IMAGES_DIR, fname)
        try:
            original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if original is None:
                print(f"Skipping {fname}: image not found.")
                continue
            original = cv2.resize(original, (IMG_SIZE, IMG_SIZE))
            pred_mask = infer_mask(model, image_path)
            pred_colored = apply_colormap(pred_mask)
            overlayed_pred = overlay_mask(original, pred_colored, alpha=0.5)
            save_path = os.path.join(OUTPUT_TEST_DIR, fname)
            cv2.imwrite(save_path, overlayed_pred)
            print(f"Saved test inference for {fname}")
            submission_dict[fname] = pred_mask.flatten()
        except Exception as e:
            print(f"Error processing {fname}: {e}")
    
    # Create and save submission CSV
    if submission_dict:
        def numeric_key(fname):
            m = re.search(r'(\d+)', fname)
            return int(m.group(1)) if m else float('inf')
        ordered_files = sorted(submission_dict.keys(), key=numeric_key)
        num_pixels = IMG_SIZE * IMG_SIZE
        rows = [f"Pixel {i}" for i in range(num_pixels)]
        submission_df = pd.DataFrame({fname: submission_dict[fname] for fname in ordered_files}, index=rows)
        submission_csv_path = os.path.join(DATA_FOLDER, "submission.csv")
        submission_df.to_csv(submission_csv_path, index=True)
        print(f"Saved submission CSV at {submission_csv_path}")

# ----------------------------
# Main Inference Execution
# ----------------------------
if __name__ == "__main__":
    # Specify the path to your trained model checkpoint
    CUSTOM_MODEL_PATH = "models/best_model_swinunetr_epoch=114_val_loss=1.3134.ckpt"  # <--- UPDATE THIS PATH
    model = load_model(CUSTOM_MODEL_PATH)
    print("Model loaded and set to evaluation mode.")
    
    print("Running inference on train images...")
    inference_on_train(model)
    
    print("Running inference on test images...")
    inference_on_test(model)
