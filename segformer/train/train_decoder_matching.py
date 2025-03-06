import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pytorch_lightning as pl

# ----------------------------
# 1. Load the ground truth masks CSV
# ----------------------------
# Read CSV, transpose so that each row corresponds to one image,
# and filter out images whose masks contain only 0s.
mask_csv_path = "y_train.csv"
mask_df = pd.read_csv(mask_csv_path, index_col=0).T
MAX_ITEMS = 55  # total number of classes (0 to 54)
mask_filter = ~(mask_df.values == 0).all(axis=-1)
mask_df = mask_df[mask_filter]
print("Filtered mask data shape:", mask_df.shape)  # e.g. (759, 65536)

# ----------------------------
# 2. Define Dataset for Decoder Training
# ----------------------------
class FeatureMaskDataset(Dataset):
    def __init__(self, features_dir, mask_df):
        """
        features_dir: directory containing .npy files of extracted features.
        mask_df: DataFrame with index as image filename (e.g., "i.png") and 65536 columns.
        """
        self.features_dir = features_dir
        self.file_names = list(mask_df.index)  # each file name corresponds to an image (e.g., "i.png")
        self.mask_df = mask_df

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        # Get the image file name (e.g., "i.png")
        file_name = self.file_names[idx]
        # Derive base name (e.g., "i") to locate feature file in extracted_features folder
        base_name, _ = os.path.splitext(file_name)
        feature_path = os.path.join(self.features_dir, f"{base_name}.npy")
        # Load pre-extracted feature map (assumed shape: (C, 256, 256))
        feature = np.load(feature_path)
        feature_tensor = torch.tensor(feature, dtype=torch.float32)
        
        # Get ground truth mask for this image from the CSV.
        # The CSV row is a flattened 256x256 mask, so reshape to (256, 256)
        mask_flat = self.mask_df.loc[file_name].values.astype(np.int64)  # shape: (65536,)
        mask = mask_flat.reshape(256, 256)
        mask_tensor = torch.tensor(mask, dtype=torch.long)
        
        return feature_tensor, mask_tensor

# ----------------------------
# 3. Define a U-Net Style Decoder Model
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
        # We assume input features are at full resolution (256x256)
        # Here, we build an "expansive" decoder that refines these features.
        self.block1 = ConvBlock(in_channels, 128)
        self.block2 = ConvBlock(128, 64)
        self.block3 = ConvBlock(64, 32)
        self.out_conv = nn.Conv2d(32, num_classes, kernel_size=1)
        
    def forward(self, x):
        # x: (B, in_channels, 256, 256)
        x = self.block1(x)  # -> (B, 128, 256, 256)
        x = self.block2(x)  # -> (B, 64, 256, 256)
        x = self.block3(x)  # -> (B, 32, 256, 256)
        out = self.out_conv(x)  # -> (B, num_classes, 256, 256)
        return out

# ----------------------------
# 4. Define Loss Functions
# ----------------------------
criterion_ce = nn.CrossEntropyLoss()  # Standard cross-entropy loss

def compute_dice_loss(pred_probs, target, num_classes, ignore_index=0, eps=1e-6):
    """
    Computes average Dice loss for classes except the ignore_index.
    pred_probs: tensor of shape (B, num_classes, H, W) from softmax.
    target: tensor of shape (B, H, W) with class indices.
    """
    dice_loss = 0.0
    count = 0
    for c in range(num_classes):
        if c == ignore_index:
            continue
        pred_c = pred_probs[:, c, :, :]
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        dice = (2. * intersection + eps) / (union + eps)
        dice_loss += (1. - dice)
        count += 1
    return dice_loss / count if count > 0 else dice_loss

def combined_loss(logits, target, num_classes):
    """
    Compute combined loss: Cross Entropy + Dice Loss.
    logits: tensor of shape (B, num_classes, 256, 256)
    target: tensor of shape (B, 256, 256) with class indices.
    """
    loss_ce = criterion_ce(logits, target)
    pred_probs = torch.softmax(logits, dim=1)
    loss_dice = compute_dice_loss(pred_probs, target, num_classes, ignore_index=0)
    return loss_ce + loss_dice, loss_ce, loss_dice

# ----------------------------
# 5. Define PyTorch Lightning Module for Decoder Training
# ----------------------------
class DecoderLightning(pl.LightningModule):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # Use the U-Net style decoder defined above.
        self.decoder = UNetDecoder(in_channels, num_classes)
        self.learning_rate = 1e-3
        self.num_classes = num_classes

    def forward(self, x):
        return self.decoder(x)

    def training_step(self, batch, batch_idx):
        features, masks = batch  # features: (B, in_channels, 256,256); masks: (B, 256,256)
        logits = self(features)  # output: (B, num_classes, 256,256)
        loss, loss_ce, loss_dice = combined_loss(logits, masks, self.num_classes)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_ce_loss", loss_ce, prog_bar=True, on_epoch=True)
        self.log("train_dice_loss", loss_dice, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.7, patience=10, min_lr=1e-6, verbose=True
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}

# ----------------------------
# 6. Main Training Script for Decoder
# ----------------------------
def main():
    # Directory containing extracted feature files (from feature extraction script)
    features_dir = "extracted_features"
    # Load the filtered CSV with ground truth masks (flattened, 65536 columns)
    mask_df = pd.read_csv("y_train.csv", index_col=0).T
    mask_filter = ~(mask_df.values == 0).all(axis=-1)
    mask_df = mask_df[mask_filter]
    print("Filtered mask data shape:", mask_df.shape)  # e.g., (759, 65536)

    # Create dataset: each sample is a (feature, mask) pair.
    dataset = FeatureMaskDataset(features_dir, mask_df)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

    # Determine number of channels from one sample feature file.
    sample_feature_path = os.path.join(features_dir, os.listdir(features_dir)[0])
    sample_feature = np.load(sample_feature_path)
    in_channels = sample_feature.shape[0]  # should be 55
    num_classes = 55  # classes: 0...54

    # Initialize the decoder Lightning module
    model = DecoderLightning(in_channels=in_channels, num_classes=num_classes)
    
    # Initialize the PyTorch Lightning Trainer (using GPU if available)
    trainer = pl.Trainer(
        max_epochs=2000,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[0],
        num_sanity_val_steps=0
    )
    trainer.fit(model, dataloader)
    
    # Save the trained decoder model to a checkpoint named "decoder.ckpt"
    save_path = "models/decoder.ckpt"
    trainer.save_checkpoint(save_path)
    print(f"Saved trained decoder checkpoint to {save_path}")

if __name__ == "__main__":
    main()
