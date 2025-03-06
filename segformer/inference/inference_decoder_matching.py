import os
import cv2
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from PIL import Image
from torchvision import transforms
from transformers import SegformerForSemanticSegmentation, SegformerConfig
from torch import nn

# ----------------------------
# Define the Backbone (SegFormer) Model for Feature Extraction
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
        # Initialize the model; weights will be loaded from checkpoint.
        self.backbone = SegformerForSemanticSegmentation(config)

    def forward(self, img):
        # Expects img of shape (B, 1, 256, 256)
        img = img / 255.0  # normalize
        out = self.backbone(pixel_values=img)[0]  # raw logits; shape: (B, 55, H_feat, W_feat)
        # Upsample to (256,256)
        out = torch.nn.functional.interpolate(out, size=(256, 256), mode="bilinear", align_corners=False)
        return out  # shape: (B, 55, 256, 256)

# ----------------------------
# Define a U-Net Style Decoder Model
# ----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNetDecoder(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(UNetDecoder, self).__init__()
        # Build a U-Net style decoder that processes features at full resolution (256x256)
        self.block1 = ConvBlock(in_channels, 128)
        self.block2 = ConvBlock(128, 64)
        self.block3 = ConvBlock(64, 32)
        self.out_conv = nn.Conv2d(32, num_classes, kernel_size=1)
        
    def forward(self, x):
        # x: (B, in_channels, 256,256)
        x = self.block1(x)  # -> (B, 128, 256,256)
        x = self.block2(x)  # -> (B, 64, 256,256)
        x = self.block3(x)  # -> (B, 32, 256,256)
        out = self.out_conv(x)  # -> (B, num_classes, 256,256)
        return out

# ----------------------------
# Preprocessing function for test images
# ----------------------------
def preprocess_image(image_path):
    """
    Read image in grayscale using OpenCV, resize to 256x256,
    convert to float32, and add batch and channel dimensions.
    Output shape: (1, 1, 256, 256)
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.float32)
    tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0)
    return tensor

# ----------------------------
# Inference helper: Run Backbone and Decoder
# ----------------------------
def run_inference(backbone, decoder, image_path, device):
    """
    Preprocess an image, extract features with the backbone, then pass them through
    the decoder to produce final logits. Argmax over channels gives the predicted mask.
    Returns: predicted mask as a numpy array of shape (256,256).
    """
    img_tensor = preprocess_image(image_path).to(device)  # (1, 1, 256,256)
    with torch.no_grad():
        features = backbone(img_tensor)        # (1, 55, 256,256)
        logits = decoder(features)             # (1, 55, 256,256)
        pred_mask = torch.argmax(logits, dim=1).squeeze(0)  # (256,256)
    return pred_mask.cpu().numpy()

# ----------------------------
# Main Inference Pipeline
# ----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the SegFormer backbone from checkpoint
    backbone_ckpt = "models/model.ckpt"
    backbone_model = MyModel(in_channels=1)
    backbone_checkpoint = torch.load(backbone_ckpt, map_location=device)
    backbone_model.load_state_dict(backbone_checkpoint["state_dict"])
    backbone_model.to(device)
    backbone_model.eval()
    
    # Load the trained decoder from its checkpoint
    decoder_ckpt = "models/decoder.ckpt"
    in_channels = 55   # as defined by MAX_ITEMS
    num_classes = 55   # classes 0...54
    decoder = UNetDecoder(in_channels=in_channels, num_classes=num_classes)
    decoder_checkpoint = torch.load(decoder_ckpt, map_location=device)
    # If the checkpoint was saved from a Lightning module, keys might be prefixed with "decoder."
    # Extract keys that belong to the decoder.
    decoder_state = {k.replace("decoder.", ""): v for k, v in decoder_checkpoint["state_dict"].items() if k.startswith("decoder.")}
    decoder.load_state_dict(decoder_state)
    decoder.to(device)
    decoder.eval()
    
    # Directories for test images and saving predictions
    test_dir = "test-images"
    predictions_dir = "test-predictions"
    os.makedirs(predictions_dir, exist_ok=True)
    
    predictions_dict = {}
    num_pixels = None  # will be set from first image (should be 256*256 = 65536)
    
    # Get sorted list of test image filenames
    image_files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))])
    
    for file_name in image_files:
        image_path = os.path.join(test_dir, file_name)
        print(f"Processing {file_name} ...")
        
        # Run inference: pass image through backbone and decoder.
        pred_mask = run_inference(backbone_model, decoder, image_path, device)  # (256,256)
        
        # Save predicted mask as an image file (convert to uint8)
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
    # Create DataFrame where each column corresponds to an image prediction
    df = pd.DataFrame(predictions_dict, index=row_names)
    submission_csv = "submission.csv"
    df.to_csv(submission_csv, index=True)
    print(f"Saved predictions to {submission_csv}")

if __name__ == "__main__":
    main()
