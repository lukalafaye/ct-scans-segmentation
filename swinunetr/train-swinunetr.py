#!/usr/bin/env python3
import os
import re
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import albumentations as A
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import StratifiedKFold
from monai.networks.nets import SwinUNETR  # MONAI's implementation

# ----------------------------
# Step 1: Configuration and Data Paths
# ----------------------------
DATA_FOLDER = "../data"  # base folder (no trailing slash)
TRAIN_IMAGES_DIR = os.path.join(DATA_FOLDER, "train-images")
Y_TRAIN_CSV = os.path.join(DATA_FOLDER, "y_train.csv")
VAL_PREDICTIONS_DIR = os.path.join(DATA_FOLDER, "val_predictions")
os.makedirs(VAL_PREDICTIONS_DIR, exist_ok=True)

IMG_SIZE = 256      # our images are 256x256
NUM_CLASSES = 55    # 0 = background, 1..54 = organs
BATCH_SIZE = 4
N_EPOCHS = 5000
MODEL_NAME = "nvidia/mit-b4"
seed = 42

# ----------------------------
# Step 2: Load and Filter CSV
# ----------------------------
df = pd.read_csv(Y_TRAIN_CSV, index_col=0).T
print("Original y_train.csv shape:", df.shape)
mask = ~(df.values == 0).all(axis=1)
df_filtered = df[mask]
print("Filtered train data shape:", df_filtered.shape)
# (df_filtered.index should have filenames like "23.png", etc.)

# ----------------------------
# Step 3: Define Data Augmentations and Dataset
# ----------------------------

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=10, border_mode=cv2.BORDER_REFLECT_101, p=0.5),  # small rotation ±10°
    A.ShiftScaleRotate(
        shift_limit=0.05,      # small translation (5%)
        scale_limit=0.05,      # small scaling (5%)
        rotate_limit=0,        # we already use Rotate above, so no extra rotation here
        border_mode=cv2.BORDER_REFLECT_101,
        p=0.5
    ),
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),  # mild gamma correction, range from 80 to 120
    # Optionally, add a very slight brightness/contrast adjustment if needed:
    # A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.3)
])

class SegmentationDataset(Dataset):
    """
    For each filename in the filtered CSV, load the corresponding grayscale image
    (from TRAIN_IMAGES_DIR) and its flattened mask (from CSV) reshaped to (IMG_SIZE, IMG_SIZE).
    The mask is then converted to one-hot encoding of shape (NUM_CLASSES, IMG_SIZE, IMG_SIZE).
    """
    def __init__(self, df, transform=None):
        self.df = df
        self.filenames = list(df.index)  # e.g. "23.png"
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image_path = os.path.join(TRAIN_IMAGES_DIR, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
        mask_flat = self.df.loc[filename].values.astype(np.uint8)
        if mask_flat.shape[0] != IMG_SIZE * IMG_SIZE:
            raise ValueError(f"Mask for {filename} has {mask_flat.shape[0]} elements; expected {IMG_SIZE*IMG_SIZE}")
        mask = mask_flat.reshape(IMG_SIZE, IMG_SIZE)
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        # One-hot encode the mask (channel order: 0=background, 1..NUM_CLASSES-1)
        one_hot_mask = np.zeros((NUM_CLASSES, IMG_SIZE, IMG_SIZE), dtype=np.float32)
        for c in range(NUM_CLASSES):
            one_hot_mask[c] = (mask == c).astype(np.float32)
        image_tensor = torch.tensor(image).unsqueeze(0)   # shape: (1, IMG_SIZE, IMG_SIZE)
        mask_tensor = torch.tensor(one_hot_mask)           # shape: (NUM_CLASSES, IMG_SIZE, IMG_SIZE)
        return image_tensor, mask_tensor

# ----------------------------
# Step 4: Stratified Split for Train/Validation
# ----------------------------
def compute_label_vector(row, num_classes=NUM_CLASSES):
    labels = np.unique(row.values)
    vec = np.zeros(num_classes, dtype=int)
    for lbl in labels:
        if lbl != 0:
            vec[int(lbl)] = 1
    return vec

multi_labels = np.stack([compute_label_vector(df_filtered.loc[name]) for name in df_filtered.index])
stratify_labels = multi_labels.sum(axis=1)
mskf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
train_indices, val_indices = next(mskf.split(np.zeros(len(df_filtered)), stratify_labels))
df_train = df_filtered.iloc[train_indices]
df_val = df_filtered.iloc[val_indices]
print("Train split shape:", df_train.shape)
print("Validation split shape:", df_val.shape)

train_dataset = SegmentationDataset(df_train, transform=transform)
val_dataset = SegmentationDataset(df_val, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=12, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ----------------------------
# Step 5: Define Combined Loss (compute Dice over all channels)
# ----------------------------
class CombinedLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(CombinedLoss, self).__init__()
        self.ce_loss = nn.functional.cross_entropy
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        inputs: (B, NUM_CLASSES, H, W) raw logits.
        targets: (B, NUM_CLASSES, H, W) one-hot encoded ground truth.
        """
        B, C, H, W = inputs.shape
        target_indices = torch.argmax(targets, dim=1)  # shape: (B, H, W)
        ce = self.ce_loss(inputs, target_indices)
        
        pred_probs = torch.softmax(inputs, dim=1)  # (B, NUM_CLASSES, H, W)
        pred_probs = pred_probs.view(B, C, -1)
        targets_flat = targets.view(B, C, -1)
        intersection = (pred_probs * targets_flat).sum(dim=2)
        union = pred_probs.sum(dim=2) + targets_flat.sum(dim=2)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()  # average over all channels and batch
        total_loss = ce + dice_loss
        return ce, dice_loss, total_loss

# ----------------------------
# Step 6: Define the SwinUNETR Model (2D)
# ----------------------------
from monai.networks.nets import SwinUNETR

model = SwinUNETR(
    img_size=(IMG_SIZE, IMG_SIZE),
    in_channels=1,
    out_channels=NUM_CLASSES,
    depths=(2, 2, 2, 2),
    num_heads=(3, 6, 12, 24),
    feature_size=96,
    norm_name="instance",
    drop_rate=0.0,
    attn_drop_rate=0.0,
    dropout_path_rate=0.0,
    normalize=True,
    spatial_dims=2
)

# ----------------------------
# Step 7: Define Additional Callbacks
# ----------------------------

# Callback to save validation predictions every 5 epochs
class ValPredictionCallback(Callback):
    def __init__(self, val_dataloader, output_dir, img_size=IMG_SIZE):
        super().__init__()
        self.val_dataloader = val_dataloader
        self.output_dir = output_dir
        self.img_size = img_size
        os.makedirs(self.output_dir, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % 5 != 0:
            return
        epoch_dir = os.path.join(self.output_dir, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        device = pl_module.device
        pl_module.model.eval()
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(self.val_dataloader):
                x = x.to(device)  # (B, 1, IMG_SIZE, IMG_SIZE)
                logits = pl_module.model(x)  # (B, NUM_CLASSES, IMG_SIZE, IMG_SIZE)
                preds = torch.argmax(logits, dim=1).cpu().numpy()  # (B, IMG_SIZE, IMG_SIZE)
                # For each sample in the batch, visualize and save
                for i in range(x.shape[0]):
                    # Convert input image to uint8 for display
                    orig = (x[i, 0].cpu().numpy() * 255).astype(np.uint8)
                    # Ground truth: convert one-hot to label indices
                    gt = torch.argmax(y[i], dim=0).cpu().numpy()
                    pred = preds[i]
                    # Apply a colormap to ground truth and prediction
                    def apply_colormap(mask):
                        mask_uint8 = np.uint8(mask)
                        scale = 255 // (NUM_CLASSES - 1) if NUM_CLASSES > 1 else 255
                        mask_scaled = mask_uint8 * scale
                        return cv2.applyColorMap(mask_scaled, cv2.COLORMAP_JET)
                    gt_colored = apply_colormap(gt)
                    pred_colored = apply_colormap(pred)
                    # Overlay the prediction on the original image
                    def overlay_mask(image, colored_mask, alpha=0.5):
                        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                        return cv2.addWeighted(image_bgr, 1 - alpha, colored_mask, alpha, 0)
                    overlayed = overlay_mask(orig, pred_colored, alpha=0.5)
                    # Combine original image, ground truth, and prediction overlay side-by-side
                    combined = np.concatenate([cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR), gt_colored, overlayed], axis=1)
                    save_path = os.path.join(epoch_dir, f"val_{batch_idx}_{i}.png")
                    cv2.imwrite(save_path, combined)
        pl_module.model.train()

# ModelCheckpoint callback to save best model every 5 epochs
checkpoint_callback = ModelCheckpoint(
    dirpath="models/",
    filename="best_model_swinunetr_{epoch:02d}_{val_loss:.4f}",
    monitor="val_loss",
    mode="min",
    save_top_k=1,
    every_n_epochs=5,
    save_weights_only=True,
    verbose=True,
)

# ----------------------------
# Step 8: Define PyTorch Lightning Module for Training
# ----------------------------
class SwinUNETRLightning(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-4):
        super().__init__()
        self.model = model
        self.loss_fn = CombinedLoss()
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch  # x: (B, 1, IMG_SIZE, IMG_SIZE), y: (B, NUM_CLASSES, IMG_SIZE, IMG_SIZE)
        logits = self(x)
        ce_loss, dice_loss, loss = self.loss_fn(logits, y)
        dice_score = 1 - dice_loss
        self.log("train_ce_loss", ce_loss, prog_bar=True, on_epoch=True)
        self.log("train_dice_loss", dice_loss, prog_bar=True, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_dice_score", dice_score, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        ce_loss, dice_loss, loss = self.loss_fn(logits, y)
        dice_score = 1 - dice_loss
        self.log("val_ce_loss", ce_loss, prog_bar=True, on_epoch=True)
        self.log("val_dice_loss", dice_loss, prog_bar=True, on_epoch=True)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_dice_score", dice_score, prog_bar=True, on_epoch=True)
        return {"val_loss": loss, "val_ce_loss": ce_loss, "val_dice_score": dice_score}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.7, patience=10, min_lr=1e-6, verbose=True
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

# ----------------------------
# Step 9: Training
# ----------------------------
logger = TensorBoardLogger("tb_logs", name="swinunetr")
model_lightning = SwinUNETRLightning(model=model, learning_rate=1e-4)

trainer = pl.Trainer(
    max_epochs=N_EPOCHS,
    accelerator="gpu",
    devices=1,
    callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=500),
               checkpoint_callback,
               ValPredictionCallback(val_dataloader=val_loader, output_dir=VAL_PREDICTIONS_DIR, img_size=IMG_SIZE)
              ],
    logger=logger,
    log_every_n_steps=10,
)

trainer.fit(model_lightning, train_loader, val_loader)
