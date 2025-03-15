import pandas as pd
from pathlib import Path
import cv2
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import sklearn.metrics
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation, SegformerConfig, SegformerModel
import torch
import random
from scipy.optimize import linear_sum_assignment
import albumentations as A
from torch.utils.data import DataLoader
import os
import gc
from torchmetrics import MeanMetric
from sklearn.model_selection import StratifiedKFold, KFold
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from scipy.spatial.distance import cdist

# Set random seed for reproducibility
seed = 12458
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

TRAIN = True
N_EPOCHS = 500
BATCH_SIZE = 4
MODEL_NAME = "nvidia/mit-b4"
DATA_FOLDER = "../data" # no /

# Load the train labels; note the transpose!
train_df = pd.read_csv(f"{DATA_FOLDER}/y_train.csv", index_col=0).T
print(train_df.shape)
MAX_ITEMS = 55
mask = ~(train_df.values == 0).all(axis=-1)
train_df = train_df[mask]
print("Train data shape:", train_df.shape)

import json

# Load the JSON file containing annotated labels
with open('../data/annotated_labels.json', 'r') as f:
    annotated_labels = json.load(f)

# Define augmentations using Albumentations
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    # Slight random rotations (±10°) with reflection to avoid black borders
    A.ShiftScaleRotate(
        shift_limit=0.0625,  # shift up to ~6%
        scale_limit=0.1,     # scale changes of 10%
        rotate_limit=10,     # rotation up to 10 degrees
        border_mode=cv2.BORDER_REFLECT_101, 
        p=0.5
    ),
    # Adjust brightness and contrast moderately.
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5)
])

# Dataset definition: fixed order one-hot encoding
class MyDataset():
    def __init__(self, train_df, annotated_labels) -> None:
        self.paths = list(train_df.index)
        self.targets = train_df.values.astype(np.uint8)
        self.annotated_labels = annotated_labels  # list of lists from JSON
        # No additional initialization needed

    def __getitem__(self, i):
        path = f"{DATA_FOLDER}/train-images/{self.paths[i]}"
        x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # (256, 256)
        y = self.targets[i].reshape(256, 256)         # (256, 256)
        # Apply augmentation
        transformed = transform(image=x, mask=y)
        x = transformed['image']
        y = transformed['mask']
        # x: (1, 256, 256) and y: one-hot encoded tensor with fixed channel order (MAX_ITEMS, 256, 256)

        img_id = int(os.path.splitext(self.paths[i])[0])
        expected = self.annotated_labels[img_id]  # e.g. [1, 34, 35, 54]

        return x[np.newaxis], self.one_hot(y), expected
    
    def one_hot(self, y):
        # Ensure y is 2D
        y = np.squeeze(y)
        H, W = y.shape
        one_hot_mask = np.zeros((MAX_ITEMS, H, W), dtype=np.float32)
        for c in range(MAX_ITEMS):
            one_hot_mask[c] = (y == c).astype(np.float32)
        return one_hot_mask
    
    def __len__(self):
        return len(self.paths)

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

def compute_label_vector(row, max_items=55):
    labels = np.unique(row.values)
    vec = np.zeros(max_items, dtype=int)
    for lbl in labels:
        if lbl != 0:
            vec[int(lbl)] = 1
    return vec

multi_labels = np.stack([compute_label_vector(train_df.loc[name]) for name in train_df.index])
mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
train_indices, val_indices = next(mskf.split(np.zeros(len(train_df)), multi_labels))
train_df_split = train_df.iloc[train_indices]
val_df_split = train_df.iloc[val_indices]

print("Train split shape:", train_df_split.shape)
print("Validation split shape:", val_df_split.shape)

train_dataset = MyDataset(train_df_split, annotated_labels)
val_dataset = MyDataset(val_df_split, annotated_labels)

print("Train split contains: ", len(train_df_split.stack().unique()),
      " - Val split contains: ", len(val_df_split.stack().unique()))

class CombinedLoss(torch.nn.Module):
    def __init__(self, smooth=1e-10):
        super(CombinedLoss, self).__init__()
        self.ce_loss = torch.nn.functional.cross_entropy
        self.smooth = smooth

    def forward(self, inputs, targets, expected_labels):
        # inputs: (B, MAX_ITEMS, 256,256) raw logits
        # targets: (B, MAX_ITEMS, 256,256) one-hot encoded ground truth
        # expected_labels: list of lists (length B), each with expected organ labels (excluding background)
        B, C, H, W = inputs.shape
        
        # --- Cross Entropy Loss ---
        # Convert one-hot targets to indices: shape (B, 256,256)
        target_indices = torch.argmax(targets, dim=1)
        ce = self.ce_loss(inputs, target_indices)
        
        # --- Dice Loss ---
        # Convert logits to probabilities (this is the crucial fix)
        probs_dice = torch.softmax(inputs, dim=1)  # shape: (B, MAX_ITEMS, 256,256)
        
        # Remove background channel (assumed to be channel 0)
        targets_dice = targets[:, :, :, :]    # shape: (B, MAX_ITEMS-1, 256,256)
        
        # Flatten spatial dimensions
        B, C_dice, H, W = probs_dice.shape
        probs_flat = probs_dice.view(B, C_dice, -1)   # (B, C_dice, H*W)
        targets_flat = targets_dice.view(B, C_dice, -1)
        
        # Compute intersection and union for dice coefficient
        intersection = (probs_flat * targets_flat).sum(dim=2)  # (B, C_dice)
        union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)  # (B, C_dice)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)  # (B, C_dice)
        
        # Create a mask: for each sample i and channel j (organ label j+1),
        # if the organ is expected (per JSON) but the target has zero pixels, exclude that channel.
        mask = torch.ones((B, C_dice), device=inputs.device)
        target_sums = targets_flat.sum(dim=2)  # (B, C_dice)
        for i in range(B):
            for j in range(C_dice):
                organ_label = j  # channels 0...C_dice-1 correspond to labels 1...MAX_ITEMS-1
                if (organ_label in expected_labels[i]) and (target_sums[i, j] == 0):
                    mask[i, j] = 0.0
        
        # Compute dice loss only over the channels not masked out
        dice_losses = []
        for i in range(B):
            if mask[i].sum() > 0:
                sample_dice = dice[i] * mask[i]
                sample_loss = 1 - (sample_dice.sum() / mask[i].sum())
                dice_losses.append(sample_loss)
            else:
                dice_losses.append(torch.tensor(0.0, device=inputs.device))
        dice_loss = torch.stack(dice_losses).mean()
        
        combined_loss = 0.4 * ce + 0.6 * dice_loss
        return ce, dice_loss, combined_loss


# LightningModule definition using the CombinedLoss
class MyLightningModule(pl.LightningModule):
    def __init__(self, in_channels=1):
        super().__init__()
        config = SegformerConfig.from_pretrained(MODEL_NAME)
        config.num_channels = in_channels
        config.id2label = {i: i for i in range(MAX_ITEMS)}
        config.label2id = {i: i for i in range(MAX_ITEMS)}
        self.config = config
        self.backbone = SegformerForSemanticSegmentation(config)
        self.loss_fn = CombinedLoss()
        self.mean_tracker = MeanMetric()

    def forward(self, img):
        # img: (B, 1, 256,256)
        img = img / 255
        out = self.backbone(pixel_values=img)[0]  # e.g., (B, MAX_ITEMS, 64,64)
        out = torch.nn.functional.interpolate(out, mode='bilinear', scale_factor=4)  # (B, MAX_ITEMS, 256,256)
        return out

    def training_step(self, batch, batch_idx):
        x, y, expected = batch  # x: (B, 1, 256,256); y: (B, MAX_ITEMS, 256,256)
        y_hat = self(x)  # (B, MAX_ITEMS, 256,256)
        ce_loss, dice_loss, loss = self.loss_fn(y_hat, y, expected)
        dice_score = 1 - dice_loss
        self.log("train_ce_loss", ce_loss, prog_bar=True, on_epoch=True)
        self.log("train_dice_loss", dice_loss, prog_bar=True, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_dice_score", dice_score, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, expected = batch
        y_hat = self(x)
        ce_loss, dice_loss, loss = self.loss_fn(y_hat, y, expected)
        dice_score = 1 - dice_loss
        self.log("val_ce_loss", ce_loss, prog_bar=True, on_epoch=True)
        self.log("val_dice_loss", dice_loss, prog_bar=True, on_epoch=True)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_dice_score", dice_score, prog_bar=True, on_epoch=True)
        return {"val_loss": loss, "val_ce_loss": ce_loss, "val_dice_score": dice_score}

    def on_train_epoch_end(self):
        self.mean_tracker.reset()

    def on_validation_epoch_end(self):
        pass

    def configure_optimizers(self):
        LR = 1e-4
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LR)
        self.reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.7, patience=10, min_lr=1e-6, verbose=True
        )
        return {"optimizer": self.optimizer, "lr_scheduler": self.reduce_lr_on_plateau, "monitor": "train_loss"}

# Set up TensorBoard logger for online logging
from pytorch_lightning.loggers import TensorBoardLogger
logger = TensorBoardLogger("tb_logs", name="json-segformer-dice-ce-all")

# Device selection and DataLoader creation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=4,
    pin_memory=True,
    shuffle=True
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=BATCH_SIZE,
    num_workers=4,
    pin_memory=True,
    shuffle=False
)

def apply_colormap(mask):
    """
    Convert a single-channel mask to a color image using a colormap.
    Assumes mask values in 0-54.
    """
    # Convert mask to uint8 if not already
    mask_uint8 = np.uint8(mask)
    colored = cv2.applyColorMap(mask_uint8 * (255 // MAX_ITEMS), cv2.COLORMAP_JET)
    return colored

def overlay_mask_on_image(image, mask, alpha=0.5):
    """
    Overlay a colored mask on the original grayscale image.
    `image`: grayscale image of shape (H, W)
    `mask`: colored mask of shape (H, W, 3)
    """
    # Convert grayscale image to BGR
    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(image_bgr, 1 - alpha, mask, alpha, 0)
    return overlay

class ValidationVisualizationCallback(pl.Callback):
    def __init__(self, val_dataloader, output_dir="generated_dice_ce"):
        super().__init__()
        self.val_dataloader = val_dataloader
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % 5 != 0:
            return
        # Create folder for this epoch
        epoch_dir = os.path.join(self.output_dir, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        
        pl_module.eval()
        device = pl_module.device
        
        # Define text area height and font parameters
        text_height = 20  # height of the text area (in pixels)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4  # very tiny text
        thickness = 1
        text_color = (0, 0, 0)
        
        # Loop over validation dataloader; optionally restrict to a subset if desired
        for batch_idx, batch in enumerate(self.val_dataloader):
            # Unpack batch; our dataset now returns (x, y, expected) but we need only x and y for visualization
            x, y, _ = batch  
            x = x.to(device)
            with torch.no_grad():
                logits = pl_module(x)  # (B, MAX_ITEMS, 256,256)
                preds = torch.argmax(logits, dim=1).cpu().numpy()  # (B, 256,256)
            # Convert ground truth from one-hot to label indices
            gt = np.argmax(y, axis=1)  # (B, 256,256)
            
            # Process each sample in the batch individually
            for i in range(x.shape[0]):
                orig = x[i, 0].cpu().numpy().astype(np.uint8)  # original grayscale image
                gt_mask = gt[i]      # ground truth mask (256,256)
                pred_mask = preds[i] # predicted mask (256,256)
                
                # Generate color maps for ground truth and prediction
                gt_color = apply_colormap(gt_mask)
                pred_color = apply_colormap(pred_mask)
                
                # Overlay masks on the original image
                gt_overlay = overlay_mask_on_image(orig, gt_color, alpha=0.5)
                pred_overlay = overlay_mask_on_image(orig, pred_color, alpha=0.5)
                
                # Convert original image to BGR for consistent visualization
                orig_bgr = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
                
                # Get unique labels (sorted) for ground truth and predicted masks
                gt_labels = sorted(np.unique(gt_mask))
                pred_labels = sorted(np.unique(pred_mask))
                # Create tiny text strings with no spaces between numbers
                gt_text = "GT:" + ",".join(map(str, gt_labels))
                pred_text = "Pred:" + ",".join(map(str, pred_labels))
                
                # Create text areas for each image section
                # For original image, we'll create a blank text area
                blank_text = np.full((text_height, orig_bgr.shape[1], 3), 255, dtype=np.uint8)
                # For the ground truth and prediction overlays, create separate text areas
                gt_text_area = np.full((text_height, gt_overlay.shape[1], 3), 255, dtype=np.uint8)
                pred_text_area = np.full((text_height, pred_overlay.shape[1], 3), 255, dtype=np.uint8)
                
                # Add tiny text onto the respective text areas
                # (x-coordinate is small so the text appears near the left edge,
                # and y is set near the bottom of the text area)
                cv2.putText(gt_text_area, gt_text, (2, text_height - 4), font, font_scale, text_color, thickness)
                cv2.putText(pred_text_area, pred_text, (2, text_height - 4), font, font_scale, text_color, thickness)
                
                # Combine the top row: original, ground truth overlay, prediction overlay
                combined_top = cv2.hconcat([orig_bgr, gt_overlay, pred_overlay])
                # Combine the text row: blank under original, then gt text, then pred text
                combined_text = cv2.hconcat([blank_text, gt_text_area, pred_text_area])
                # Finally, stack the top and text rows vertically
                final_vis = cv2.vconcat([combined_top, combined_text])
                
                # Save the visualization with a unique filename
                save_path = os.path.join(epoch_dir, f"val_{batch_idx}_{i}.png")
                cv2.imwrite(save_path, final_vis)
        
        pl_module.train()  # set model back to train mode

# Now, when you create your Trainer, add this callback:
viz_callback = ValidationVisualizationCallback(val_dataloader=val_loader, output_dir="json_generated_dice_ce")


# Main training execution
if TRAIN:
    model = MyLightningModule()
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="json_models/",
        filename="json_dice_ce_best_val",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        every_n_epochs=5,  # check and save every 5 epochs
        save_weights_only=True,
        verbose=True
    )
    stopping_callback = EarlyStopping(monitor="val_loss", mode="min", patience=50)
    trainer = pl.Trainer(
        max_epochs=N_EPOCHS,
        accelerator="gpu",
        devices=[0],
        accumulate_grad_batches=1,
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback, stopping_callback, viz_callback],
        logger=logger
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
