import os
import cv2
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from transformers import SegformerForSemanticSegmentation, SegformerConfig

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
        # Expects img of shape (B, 1, 256, 256)
        img = img / 255.0  # Normalize as in training
        out = self.backbone(pixel_values=img)[0]
        # Upsample to (256,256) if necessary (should match training)
        out = torch.nn.functional.interpolate(out, size=(256, 256), mode="bilinear", align_corners=False)
        return out

# ----------------------------
# Preprocessing function
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
    tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0)  # shape: (1, 1, 256, 256)
    return tensor

# ----------------------------
# Main Feature Extraction Script
# ----------------------------
def main():
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load filtered training labels from CSV (transposed so that each row is one image)
    train_df = pd.read_csv("y_train.csv", index_col=0).T
    MAX_ITEMS = 55
    mask = ~(train_df.values == 0).all(axis=-1)
    train_df = train_df[mask]
    print("Filtered train data shape:", train_df.shape)  # e.g., (759, 65536)
    
    # Set up model and load checkpoint
    checkpoint_path = "models/model.ckpt"
    model = MyModel(in_channels=1)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()  # Set model to evaluation mode
    
    # Directory containing training images
    images_dir = "train-images"
    # Directory to save extracted features
    output_dir = "extracted_features"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process only images that are in the filtered train_df (their index contains the image filenames, e.g., "i.png")
    for file_name in train_df.index:
        image_path = os.path.join(images_dir, file_name)
        if not os.path.exists(image_path):
            print(f"Image {file_name} not found in {images_dir}, skipping...")
            continue
        
        print(f"Processing {file_name} ...")
        # Preprocess image (read, resize, convert to tensor)
        img_tensor = preprocess_image(image_path).to(device)  # shape: (1, 1, 256, 256)
        
        # Run inference to extract features (raw logits before argmax)
        with torch.no_grad():
            features = model(img_tensor)  # shape: (1, MAX_ITEMS, 256, 256)
        
        # Remove the batch dimension and convert features to numpy array
        features_np = features.cpu().numpy()[0]
        
        # Save the feature map as a .npy file using the image filename (without extension)
        base_name, _ = os.path.splitext(file_name)
        out_path = os.path.join(output_dir, f"{base_name}.npy")
        np.save(out_path, features_np)
        print(f"Saved features for {file_name} to {out_path}")
        
        # Optional: clear GPU cache to free memory
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
