import os
import json
import torch
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import segmentation_models_pytorch as smp
import pytorch_lightning as pl

# ---------------------------
# IMPORTANT: Import your UNetLightning class and 
# any other custom pieces from the training script.
# Example:
from fullsm import UNetLightning  # <-- Make sure this import path is correct

# -------------- USER CONFIG --------------
DATA_FOLDER = "../data"
TRAIN_IMAGES_DIR = os.path.join(DATA_FOLDER, "train-images")
TEST_IMAGES_DIR  = os.path.join(DATA_FOLDER, "test-images")
Y_TRAIN_CSV      = os.path.join(DATA_FOLDER, "y_train.csv")

OUTPUT_TRAIN_DIR = os.path.join(DATA_FOLDER, "allb4inference_train_output")
OUTPUT_TEST_DIR  = os.path.join(DATA_FOLDER, "allb4inference_test_output")
SUBMISSION_FILE  = os.path.join(DATA_FOLDER, "submissionb4all.csv")

ANNOTATED_LABELS_JSON = os.path.join(DATA_FOLDER, "annotated_labels.json")
CHECKPOINT_PATH = "models/.....131.ckpt"

IMG_SIZE = 256
NUM_CLASSES = 55
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_TRAIN_DIR, exist_ok=True)
os.makedirs(OUTPUT_TEST_DIR,  exist_ok=True)
# -----------------------------------------

def extract_number(filename):
    """Extract numeric portion from a filename like "12.png" => 12."""
    return int(os.path.splitext(filename)[0])


def load_model(checkpoint_path, device="cpu"):
    """
    Properly load the trained UNetLightning from the checkpoint
    by reconstructing the same SMP model and passing it to load_from_checkpoint.
    """
    print(f"Loading model from {checkpoint_path}")

    # Rebuild EXACT SMP model you used in training:
    base_model = smp.Unet(
        encoder_name="efficientnet-b4",    # same as training
        encoder_weights="imagenet",        # same as training
        in_channels=1,                     # same as training
        classes=NUM_CLASSES                # same as training (55)
    )

    # Provide ALL required __init__ args to match your training code
    lightning_model = UNetLightning.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model=base_model,           # <-- required
        learning_rate=3e-3,         # or the value you used
        ce_weight=0.5,
        dice_weight=0.5,
        map_location=device
    )
    lightning_model.eval()
    lightning_model.to(device)
    return lightning_model

def run_inference(lightning_model, image_path, device="cpu"):
    """
    Loads and preprocesses a single image, runs inference, returns predicted mask (H,W) as np.uint8.
    """
    # Read grayscale image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    # Convert to tensor (B,1,H,W), normalizing if that was done in training
    tensor_img = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)
    tensor_img /= 255.0

    # Forward pass
    with torch.no_grad():
        # The forward in UNetLightning calls self.model(x)
        logits = lightning_model(tensor_img)  # shape (B, C, H, W)
        pred_mask = torch.argmax(logits, dim=1)  # (B, H, W)

    return pred_mask.squeeze(0).cpu().numpy().astype(np.uint8)

def visualize_and_save(
    image_path,
    pred_mask,
    output_folder,
    prefix="",
    gt_mask=None,
    json_classes=None,
    mask_classes=None
):
    """
    Saves a visualization image:
      - If gt_mask is provided, show: [Original, GT, Predicted]
      - Otherwise, show: [Original, Predicted]
    Also displays:
      - JSON classes
      - Mask classes
      - Predicted classes
      - Missing labels (in JSON but not in predictions)
      - Incorrect labels (in predictions but not in JSON)
    """
    filename_only = os.path.basename(image_path)
    base_name = os.path.splitext(filename_only)[0]

    # Build a colormap with background=white
    cmap_colors = plt.cm.jet(np.linspace(0, 1, NUM_CLASSES))
    cmap_colors[0] = [1, 1, 1, 1]  # background => white
    cmap = ListedColormap(cmap_colors)

    # Load original image for display
    orig = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if orig is None:
        orig = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    orig = cv2.resize(orig, (IMG_SIZE, IMG_SIZE))

    # Pred classes
    pred_unique = np.unique(pred_mask)
    pred_classes_list = [c for c in pred_unique if c != 0]

    # GT classes (train images only)
    if gt_mask is not None:
        gt_unique = np.unique(gt_mask)
        gt_classes_list = [c for c in gt_unique if c != 0]
    else:
        gt_classes_list = []

    # JSON classes
    json_cls_list = sorted(list(json_classes)) if json_classes else []
    missing_labels = set(json_cls_list) - set(pred_classes_list)
    incorrect_labels = set(pred_classes_list) - set(json_cls_list)

    mask_cls_list = sorted(list(mask_classes)) if mask_classes else []

    # Plot
    if gt_mask is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(orig, cmap='gray')
        axes[0].set_title("Original")
        axes[0].axis('off')

        axes[1].imshow(gt_mask, cmap=cmap, vmin=0, vmax=NUM_CLASSES-1)
        axes[1].set_title("Ground Truth")
        axes[1].axis('off')

        axes[2].imshow(pred_mask, cmap=cmap, vmin=0, vmax=NUM_CLASSES-1)
        axes[2].set_title("Prediction")
        axes[2].axis('off')

    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(orig, cmap='gray')
        axes[0].set_title("Original")
        axes[0].axis('off')

        axes[1].imshow(pred_mask, cmap=cmap, vmin=0, vmax=NUM_CLASSES-1)
        axes[1].set_title("Prediction")
        axes[1].axis('off')

    info_str = f"JSON classes: {json_cls_list}\n"
    if gt_mask is not None:
        info_str += f"Mask classes: {mask_cls_list}\n"
    info_str += f"Pred classes: {sorted(pred_classes_list)}\n"
    info_str += f"Missing from JSON: {sorted(missing_labels)}\n"
    info_str += f"Incorrect predicted: {sorted(incorrect_labels)}\n"

    plt.suptitle(f"{prefix}{filename_only}")
    plt.figtext(0.01, 0.01, info_str, fontsize=9)

    out_path = os.path.join(output_folder, f"{prefix}{base_name}.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

def main():
    # 1) Load the model properly
    lightning_model = load_model(CHECKPOINT_PATH, device=DEVICE)

    # 2) Load y_train.csv, filter all-background
    train_df = pd.read_csv(Y_TRAIN_CSV, index_col=0).T
    mask = ~(train_df.values == 0).all(axis=-1)
    train_df = train_df[mask]

    # 3) If we have annotated_labels.json, load it (list of lists)
    annotated_labels = []
    if os.path.exists(ANNOTATED_LABELS_JSON):
        with open(ANNOTATED_LABELS_JSON, "r") as f:
            annotated_labels = json.load(f)

    # 4) Inference on training images
    train_files = [
        f for f in os.listdir(TRAIN_IMAGES_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]

    for file_name in train_files:
        if file_name not in train_df.index:
            # skip images not in train_df
            continue

        image_path = os.path.join(TRAIN_IMAGES_DIR, file_name)
        pred_mask = run_inference(lightning_model, image_path, device=DEVICE)

        # Ground truth from CSV
        gt_flat = train_df.loc[file_name].values.astype(np.uint8)
        gt_mask = gt_flat.reshape(IMG_SIZE, IMG_SIZE)

        mask_unique = np.unique(gt_mask)
        mask_classes = set(mask_unique[mask_unique != 0])

        # Convert filename to int => JSON classes
        base_idx = int(os.path.splitext(file_name)[0])
        json_cls = set()
        if 0 <= base_idx < len(annotated_labels):
            json_cls = set(annotated_labels[base_idx])
            if 0 in json_cls:
                json_cls.remove(0)

        visualize_and_save(
            image_path=image_path,
            pred_mask=pred_mask,
            output_folder=OUTPUT_TRAIN_DIR,
            prefix="train_",
            gt_mask=gt_mask,
            json_classes=json_cls,
            mask_classes=mask_classes
        )

    # 5) Inference on test images (no GT)
    predictions_dict = {}
    num_pixels = None

    test_files = [
        f for f in os.listdir(TEST_IMAGES_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]

    for file_name in test_files:
        image_path = os.path.join(TEST_IMAGES_DIR, file_name)
        pred_mask = run_inference(lightning_model, image_path, device=DEVICE)

        # Optional visualization for test
        visualize_and_save(
            image_path=image_path,
            pred_mask=pred_mask,
            output_folder=OUTPUT_TEST_DIR,
            prefix="test_",
            gt_mask=None,
            json_classes=set(),  # no GT or JSON info for test
            mask_classes=None
        )

        # Flatten predictions for CSV
        flat_pred = pred_mask.flatten()
        if num_pixels is None:
            num_pixels = flat_pred.shape[0]
        elif num_pixels != flat_pred.shape[0]:
            raise ValueError(f"Inconsistent image sizes detected for {file_name}.")

        predictions_dict[file_name] = flat_pred

    # 6) Create submission CSV with columns=filenames, rows= "Pixel 0..(num_pixels-1)"
    row_names = [f"Pixel {i}" for i in range(num_pixels)]
    submission_df = pd.DataFrame(predictions_dict, index=row_names)

    # Sort columns by numeric portion of filename
    sorted_columns = sorted(submission_df.columns, key=extract_number)
    submission_df = submission_df[sorted_columns]

    submission_df.to_csv(SUBMISSION_FILE, index=True)
    print(f"Saved sorted submission CSV to {SUBMISSION_FILE}")

if __name__ == "__main__":
    main()
