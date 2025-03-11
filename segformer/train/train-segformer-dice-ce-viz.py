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
    def __init__(self, train_df) -> None:
        self.paths = list(train_df.index)
        self.targets = train_df.values.astype(np.uint8)
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
        return x[np.newaxis], self.one_hot(y)
    
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

train_dataset = MyDataset(train_df_split)
val_dataset = MyDataset(val_df_split)

print("Train split contains: ", len(train_df_split.stack().unique()),
      " - Val split contains: ", len(val_df_split.stack().unique()))

class CombinedLoss(torch.nn.Module):
    def __init__(self, smooth=1e-10):
        super(CombinedLoss, self).__init__()
        self.ce_loss = torch.nn.functional.cross_entropy
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs: (B, MAX_ITEMS, 256,256) raw logits
        # targets: (B, MAX_ITEMS, 256,256) one-hot encoded ground truth
        B, C, H, W = inputs.shape
        
        # Cross Entropy: convert one-hot targets to indices
        target_indices = torch.argmax(targets, dim=1)  # shape: (B, 256,256)
        ce = self.ce_loss(inputs, target_indices)

        # For dice loss, first convert logits to probabilities
        probs = torch.softmax(inputs, dim=1)

        # Remove background channel (assumed channel 0)
        probs = probs[:, 1:, :, :]      # shape: (B, MAX_ITEMS-1, 256,256)
        targets = targets[:, 1:, :, :]    # shape: (B, MAX_ITEMS-1, 256,256)

        B, C_dice, H, W = probs.shape
        probs_flat = probs.view(B, C_dice, -1)   # shape: (B, C_dice, H*W)
        targets_flat = targets.view(B, C_dice, -1)

        intersection = (probs_flat * targets_flat).sum(dim=2)
        union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()  # average over channels and batch
        
        return ce, dice_loss, 0.2*ce + 0.8*dice_loss


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
        x, y = batch  # x: (B, 1, 256,256); y: (B, MAX_ITEMS, 256,256)
        y_hat = self(x)  # (B, MAX_ITEMS, 256,256)
        ce_loss, dice_loss, loss = self.loss_fn(y_hat, y)
        dice_score = 1 - dice_loss
        self.log("train_ce_loss", ce_loss, prog_bar=True, on_epoch=True)
        self.log("train_dice_loss", dice_loss, prog_bar=True, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_dice_score", dice_score, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        ce_loss, dice_loss, loss = self.loss_fn(y_hat, y)
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
logger = TensorBoardLogger("tb_logs", name="no-json-segformer-dice-ce")

# Device selection and DataLoader creation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=os.cpu_count()-4,
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

print("Train split contains: ", len(train_df_split.stack().unique()), 
      " - Val split contains: ", len(val_df_split.stack().unique()))




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
        # Get the current epoch
        epoch = trainer.current_epoch
        if epoch % 5 != 0:
            return
        # Create folder for this epoch
        epoch_dir = os.path.join(self.output_dir, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        
        pl_module.eval()
        device = pl_module.device
        
        # Loop over the validation dataloader (optionally, you may restrict to a subset)
        for batch_idx, batch in enumerate(self.val_dataloader):
            # Batch returns (x, y); x shape: (B, 1, 256, 256); y: (B, MAX_ITEMS, 256,256)
            x, y = batch
            x = x.to(device)
            with torch.no_grad():
                # Run inference
                logits = pl_module(x)  # shape: (B, MAX_ITEMS, 256,256)
                preds = torch.argmax(logits, dim=1).cpu().numpy()  # shape: (B, 256,256)
            # Convert ground truth from one-hot to label indices
            gt = np.argmax(y, axis=1)  # shape: (B, 256,256)
            # Process each sample in the batch individually
            for i in range(x.shape[0]):
                # Get original image (convert from tensor to numpy)
                orig = x[i, 0].cpu().numpy().astype(np.uint8)
                gt_mask = gt[i]  # shape: (256,256)
                pred_mask = preds[i]  # shape: (256,256)
                
                # Apply color maps
                gt_color = apply_colormap(gt_mask)
                pred_color = apply_colormap(pred_mask)
                
                # Overlay masks on original image
                gt_overlay = overlay_mask_on_image(orig, gt_color, alpha=0.5)
                pred_overlay = overlay_mask_on_image(orig, pred_color, alpha=0.5)
                
                # Get unique labels as sorted lists for ground truth and predictions
                gt_labels = np.unique(gt_mask)
                pred_labels = np.unique(pred_mask)
                gt_text = "GT: " + ", ".join(map(str, sorted(gt_labels)))
                pred_text = "Pred: " + ", ".join(map(str, sorted(pred_labels)))
                
                # Create a blank area for text
                H, W = orig.shape
                text_area = np.full((50, W, 3), 255, dtype=np.uint8)
                # Add text using cv2.putText
                cv2.putText(text_area, gt_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
                cv2.putText(text_area, pred_text, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
                
                # Combine images horizontally: original, gt overlay, pred overlay
                orig_bgr = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
                combined_top = cv2.hconcat([orig_bgr, gt_overlay, pred_overlay])
                # Add text area below the gt and pred images only (assuming same width for gt and pred)
                # Create a blank image for original text (optional)
                blank = np.full((50, orig_bgr.shape[1], 3), 255, dtype=np.uint8)
                combined_text = cv2.hconcat([blank, text_area, text_area])
                
                # Finally, combine top and text area vertically
                final_vis = cv2.vconcat([combined_top, combined_text])
                
                # Save the visualization with a unique filename
                save_path = os.path.join(epoch_dir, f"val_{batch_idx}_{i}.png")
                cv2.imwrite(save_path, final_vis)
        
        pl_module.train()  # set model back to train mode

# Now, when you create your Trainer, add this callback:
viz_callback = ValidationVisualizationCallback(val_dataloader=val_loader, output_dir="no_json_generated_dice_ce")


# Main training execution
if TRAIN:
    model = MyLightningModule()
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="no_json_models/",
        filename="no_json_dice_ce_best_val.ckpt",
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
