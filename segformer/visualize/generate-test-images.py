import os
import cv2
import numpy as np
import pandas as pd
import torch
import albumentations as A
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import pytorch_lightning as pl

# ----------------------------
# Define the Lightning Module for Inference
# (Same model configuration as during training)
# ----------------------------
class MyModel(pl.LightningModule):
    def __init__(self, in_channels=1):
        super().__init__()
        MODEL_NAME = "nvidia/mit-b4"
        MAX_ITEMS = 55
        config = SegformerConfig.from_pretrained(MODEL_NAME)
        config.num_channels = in_channels  # grayscale
        config.id2label = {i: i for i in range(MAX_ITEMS)}
        config.label2id = {i: i for i in range(MAX_ITEMS)}
        self.config = config
        # Initialize the model (weights will be loaded from a checkpoint)
        self.backbone = SegformerForSemanticSegmentation(config)

    def forward(self, img):
        # Expecting img of shape (B, 1, 256, 256)
        img = img / 255.0  # normalize
        # Forward pass (output might be at a lower resolution, e.g. 64x64)
        out = self.backbone(pixel_values=img)[0]
        # Upsample to the desired 256x256 resolution
        out = torch.nn.functional.interpolate(out, size=(256, 256), mode="bilinear", align_corners=False)
        return out

# ----------------------------
# Inference helper functions
# ----------------------------
def preprocess_image(image_path):
    """Read image in grayscale, resize to 256x256, and convert to tensor."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.float32)
    # Add channel and batch dimensions: (1, 1, 256, 256)
    tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0)
    return tensor

def run_inference(model, image_path):
    """Run model inference on a single image and return the predicted mask."""
    input_tensor = preprocess_image(image_path).to("cuda")
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)  # output shape: (1, MAX_ITEMS, 256, 256)
        # Take argmax over the channel dimension to get predicted segmentation mask.
        pred = torch.argmax(output, dim=1).squeeze(0)  # shape: (256, 256)
        pred_np = pred.cpu().numpy()
    return pred_np

# ----------------------------
# Main Inference Script
# ----------------------------
def main():
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the trained model checkpoint.
    # Adjust the checkpoint path as needed.
    checkpoint_path = "models/model.ckpt"
    model = MyModel(in_channels=1)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    
    # Directory containing test images
    test_dir = "test-images"
    # Create directory to save predicted masks
    predictions_dir = "predictions"
    os.makedirs(predictions_dir, exist_ok=True)
    
    predictions_dict = {}
    num_pixels = None
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    
    # Process each image in the test directory
    for file_name in image_files:
        image_path = os.path.join(test_dir, file_name)
        print(f"Processing {file_name} ...")
        pred_mask = run_inference(model, image_path)
        
        # Save the predicted mask as an image file (convert to uint8)
        save_path = os.path.join(predictions_dir, file_name)
        cv2.imwrite(save_path, pred_mask.astype(np.uint8))
        
        # Flatten the prediction into a 1D array (row-major order)
        flat_pred = pred_mask.flatten()
        if num_pixels is None:
            num_pixels = flat_pred.shape[0]
        elif num_pixels != flat_pred.shape[0]:
            raise ValueError("Inconsistent image sizes detected.")
        predictions_dict[file_name] = flat_pred
    
    # Create row names: "Pixel 0", "Pixel 1", ..., "Pixel {num_pixels-1}"
    row_names = [f"Pixel {i}" for i in range(num_pixels)]
    # Create a DataFrame where each column is an image prediction
    df = pd.DataFrame(predictions_dict, index=row_names)
    df.to_csv("predictions.csv", index=True)
    print("Saved predictions to predictions.csv")

if __name__ == "__main__":
    main()
