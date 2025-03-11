import re
import os
import cv2
import numpy as np
import pandas as pd
import torch
import albumentations as A
import pytorch_lightning as pl
from transformers import SegformerForSemanticSegmentation, SegformerConfig

# ----------------------------
# Configuration
# ----------------------------
DATA_FOLDER = "../data"  # no trailing /
TRAIN_IMAGES_DIR = os.path.join(DATA_FOLDER, "train-images")
TEST_IMAGES_DIR = os.path.join(DATA_FOLDER, "test-images")
OUTPUT_TRAIN_DIR = "no-json-train-inference-segformer-dice-ce"
OUTPUT_TEST_DIR = "no-json-test-inference-segformer-dice-ce"
os.makedirs(OUTPUT_TRAIN_DIR, exist_ok=True)
os.makedirs(OUTPUT_TEST_DIR, exist_ok=True)
SUBMISSION_FILE = "no-json-dice_ce.csv"

MODEL_CHECKPOINT = "../train/no_json_models/no_json_dice_ce_best_val.ckpt"
MAX_ITEMS = 55

# ----------------------------
# Define the Inference Model (same as training)
# ----------------------------
class MyLightningModule(pl.LightningModule):
    def __init__(self, in_channels=1):
        super().__init__()
        MODEL_NAME = "nvidia/mit-b4"
        config = SegformerConfig.from_pretrained(MODEL_NAME)
        config.num_channels = in_channels  # grayscale
        config.id2label = {i: i for i in range(MAX_ITEMS)}
        config.label2id = {i: i for i in range(MAX_ITEMS)}
        self.config = config
        # Initialize backbone (weights will be loaded from checkpoint)
        self.backbone = SegformerForSemanticSegmentation(config)

    def forward(self, img):
        # img: (B, 1, 256, 256)
        img = img / 255.0  # normalize
        out = self.backbone(pixel_values=img)[0]  # e.g., (B, MAX_ITEMS, 64, 64)
        # Upsample to 256x256
        out = torch.nn.functional.interpolate(out, size=(256, 256), mode="bilinear", align_corners=False)
        return out

# ----------------------------
# Preprocessing Function
# ----------------------------
def preprocess_image(image_path):
    """
    Reads an image in grayscale, resizes to 256x256, converts to float32,
    and adds batch and channel dimensions. Output shape: (1, 1, 256, 256)
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.float32)
    tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0)
    return tensor

# ----------------------------
# Visualization Helper Functions
# ----------------------------
def apply_colormap(mask):
    """
    Convert a single-channel mask (values 0..54) to a colored image.
    """
    # Scale mask values to 0-255
    mask_uint8 = np.uint8(mask)
    colored = cv2.applyColorMap(mask_uint8 * (255 // MAX_ITEMS), cv2.COLORMAP_JET)
    return colored

def overlay_mask_on_image(image, mask, alpha=0.5):
    """
    Overlay a colored mask on the original grayscale image.
    """
    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(image_bgr, 1 - alpha, mask, alpha, 0)
    return overlay

def add_text_below(image, text, height=30, font_scale=0.5, thickness=1):
    """
    Create an image with text centered horizontally.
    """
    H, W, _ = image.shape
    text_img = np.full((height, W, 3), 255, dtype=np.uint8)
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_x = (W - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    cv2.putText(text_img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness)
    return text_img

# ----------------------------
# Inference and Visualization Functions
# ----------------------------
def run_inference(model, image_path, device):
    input_tensor = preprocess_image(image_path).to(device)
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)  # (1, MAX_ITEMS, 256, 256)
        pred = torch.argmax(output, dim=1).squeeze(0)  # (256,256)
    return pred.cpu().numpy()

def visualize_and_save(image_path, pred_mask, gt_mask=None, output_folder=".", prefix=""):
    """
    Create a composite image:
    - For train images: original image, GT overlay, prediction overlay side by side.
      And below each image, add text with unique labels.
    - For test images (gt_mask is None): original and prediction.
    Save the composite image.
    """
    orig = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    orig = cv2.resize(orig, (256, 256))
    orig_bgr = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
    
    pred_color = apply_colormap(pred_mask)
    pred_overlay = overlay_mask_on_image(orig, pred_color, alpha=0.5)
    
    if gt_mask is not None:
        gt_color = apply_colormap(gt_mask)
        gt_overlay = overlay_mask_on_image(orig, gt_color, alpha=0.5)
        
        # Get text for ground truth and prediction
        gt_text = "GT: " + ", ".join(map(str, sorted(np.unique(gt_mask))))
        pred_text = "Pred: " + ", ".join(map(str, sorted(np.unique(pred_mask))))
        
        # Create text images (small text)
        gt_text_img = add_text_below(gt_overlay, gt_text, height=30, font_scale=0.5, thickness=1)
        pred_text_img = add_text_below(pred_overlay, pred_text, height=30, font_scale=0.5, thickness=1)
        
        # For original image, we can leave blank text
        blank_text = np.full((30, orig_bgr.shape[1], 3), 255, dtype=np.uint8)
        
        # Stack images horizontally: original, gt overlay, pred overlay
        top_row = cv2.hconcat([orig_bgr, gt_overlay, pred_overlay])
        text_row = cv2.hconcat([blank_text, gt_text_img, pred_text_img])
        final_vis = cv2.vconcat([top_row, text_row])
    else:
        # For test images: only original and prediction
        pred_text = "Pred: " + ", ".join(map(str, sorted(np.unique(pred_mask))))
        pred_text_img = add_text_below(pred_overlay, pred_text, height=30, font_scale=0.5, thickness=1)
        blank_text = np.full((30, orig_bgr.shape[1], 3), 255, dtype=np.uint8)
        top_row = cv2.hconcat([orig_bgr, pred_overlay])
        text_row = cv2.hconcat([blank_text, pred_text_img])
        final_vis = cv2.vconcat([top_row, text_row])
    
    # Save the composite image
    os.makedirs(output_folder, exist_ok=True)
    base_name = os.path.basename(image_path)
    save_path = os.path.join(output_folder, prefix + base_name)
    cv2.imwrite(save_path, final_vis)

# Sort the columns by the numeric part of the filename
def extract_number(filename):
    """Extract numeric part from a filename, e.g. '186.png' -> 186."""
    number_str = re.sub(r'\D', '', filename)
    return int(number_str) if number_str else float('inf')

# ----------------------------
# Main Inference Script
# ----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model from checkpoint
    model = MyLightningModule(in_channels=1)
    checkpoint = torch.load(MODEL_CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    
    # Load ground truth for training images from y_train.csv
    train_df = pd.read_csv(f"{DATA_FOLDER}/y_train.csv", index_col=0).T
    mask = ~(train_df.values == 0).all(axis=-1)
    train_df = train_df[mask]
    
    # Inference on training images (with ground truth visualization)
    train_files = [f for f in os.listdir(TRAIN_IMAGES_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
    for file_name in train_files:
        if file_name not in train_df.index:
            continue  # only process images present in train_df
        image_path = os.path.join(TRAIN_IMAGES_DIR, file_name)
        pred_mask = run_inference(model, image_path, device)
        # Get ground truth mask from CSV (flattened, reshape to 256x256)
        gt_flat = train_df.loc[file_name].values.astype(np.uint8)
        gt_mask = gt_flat.reshape(256, 256)
        visualize_and_save(image_path, pred_mask, gt_mask=gt_mask, output_folder=OUTPUT_TRAIN_DIR, prefix="train_")
    
    # Inference on test images (no ground truth)
    predictions_dict = {}
    num_pixels = None
    test_files = [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
    for file_name in test_files:
        image_path = os.path.join(TEST_IMAGES_DIR, file_name)
        pred_mask = run_inference(model, image_path, device)
        visualize_and_save(image_path, pred_mask, gt_mask=None, output_folder=OUTPUT_TEST_DIR, prefix="test_")
        flat_pred = pred_mask.flatten()
        if num_pixels is None:
            num_pixels = flat_pred.shape[0]
        elif num_pixels != flat_pred.shape[0]:
            raise ValueError("Inconsistent image sizes detected.")
        predictions_dict[file_name] = flat_pred
    
    # Create submission CSV from test predictions
    row_names = [f"Pixel {i}" for i in range(num_pixels)]
    submission_df = pd.DataFrame(predictions_dict, index=row_names)

    sorted_columns = sorted(submission_df.columns, key=extract_number)
    submission_df = submission_df[sorted_columns]

    submission_df.to_csv(SUBMISSION_FILE, index=True)
    print(f"Saved sorted submission CSV to {SUBMISSION_FILE}")

if __name__ == "__main__":
    main()
