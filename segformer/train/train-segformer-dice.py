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

# Load the train labels; note the transpose!
train_df = pd.read_csv("y_train.csv", index_col=0).T
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
    # Elastic deformation can simulate soft tissue distortions, but use it moderately.
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
    # Add Gaussian noise to simulate acquisition noise.
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    # Adjust brightness and contrast moderately.
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5)
])

# Dataset definition
class MyDataset():
    def __init__(self, train_df) -> None:
        self.paths = list(train_df.index)
        self.targets = train_df.values.astype(np.uint8)
        # No additional initialization needed

    def __getitem__(self, i):
        path = f"train-images/{self.paths[i]}"
        x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # (256, 256)
        y = self.targets[i].reshape(256, 256)         # (256, 256)
        # Apply augmentation
        transformed = transform(image=x, mask=y)
        x = transformed['image']
        y = transformed['mask']
        # x: (1, 256, 256) and y: one-hot encoded tensor (MAX_ITEMS, 256, 256)
        return x[np.newaxis], self.one_hot(y)
    
    def one_hot(self, y):
        # Ensure y is 2D (remove extra dimensions if any)
        y = np.squeeze(y)
        H, W = y.shape
        one_hot_mask = np.zeros((MAX_ITEMS, H, W), dtype=np.float32)
        for c in range(MAX_ITEMS):
            one_hot_mask[c] = (y == c).astype(np.float32)
        return one_hot_mask
    
    def __len__(self):
        return len(self.paths)

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# For each image, compute a binary vector indicating presence/absence of each class.
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

# Create separate datasets / data loaders
train_dataset = MyDataset(train_df_split)
val_dataset = MyDataset(val_df_split)

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

print("Train split contains: ", len(train_df_split.stack().unique()), " - Val split contains: ", len(val_df_split.stack().unique()))

# Average Dice Loss definition computed per channel (no matching, fixed channel order)
class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-10):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs and targets are assumed to be of shape (B, MAX_ITEMS, 256, 256)
        B, C, H, W = inputs.shape
        inputs = inputs.view(B, C, -1)    # shape: (B, C, 256*256)
        targets = targets.view(B, C, -1)    # shape: (B, C, 256*256)
        intersection = (inputs * targets).sum(dim=2)
        union = inputs.sum(dim=2) + targets.sum(dim=2)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        loss = 1 - dice.mean()  # average over all channels and batch
        return loss

# LightningModule definition (using only average Dice loss on all channels, fixed ordering)
class MyLightningModule(pl.LightningModule):
    def __init__(self, in_channels=1):
        super().__init__()
        config = SegformerConfig.from_pretrained(MODEL_NAME)
        config.num_channels = in_channels
        config.id2label = {i: i for i in range(MAX_ITEMS)}
        config.label2id = {i: i for i in range(MAX_ITEMS)}
        self.config = config
        self.backbone = SegformerForSemanticSegmentation(config)
        self.loss_fn = DiceLoss()
        self.mean_tracker = MeanMetric()

    def forward(self, img):
        # img shape: (B, 1, 256, 256)
        img = img / 255
        out = self.backbone(pixel_values=img)[0]  # (B, MAX_ITEMS, 64, 64)
        out = torch.nn.functional.interpolate(out, mode='bilinear', scale_factor=4)  # (B, MAX_ITEMS, 256, 256)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch  # x: (B, 1, 256, 256); y: (B, MAX_ITEMS, 256, 256)
        y_hat = self(x)  # (B, MAX_ITEMS, 256, 256)
        dice_loss = self.loss_fn(torch.softmax(y_hat, dim=1), y)
        loss = dice_loss
        dice_score = 1 - dice_loss
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_dice_score", dice_score, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        dice_loss = self.loss_fn(torch.softmax(y_hat, dim=1), y)
        loss = dice_loss
        dice_score = 1 - dice_loss
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_dice_score", dice_score, prog_bar=True, on_epoch=True)
        return {"val_loss": loss, "val_dice_score": dice_score}

    def on_train_epoch_end(self):
        self.mean_tracker.reset()

    def on_validation_epoch_end(self):
        pass

    def configure_optimizers(self):
        LR = 1e-4
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LR)
        self.reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.7,
            patience=10,
            min_lr=1e-6,
            verbose=True
        )
        return {"optimizer": self.optimizer, "lr_scheduler": self.reduce_lr_on_plateau, "monitor": "train_loss"}

# Set up TensorBoard logger for online logging
from pytorch_lightning.loggers import TensorBoardLogger
logger = TensorBoardLogger("tb_logs", name="segform-no-shuffle")

# Device selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN = True

if TRAIN:
    model = MyLightningModule()

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="models/",
        filename="best_model_no_shuffle",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        every_n_epochs=5,
        save_weights_only=True,
        verbose=True
    )

    stopping_callback = EarlyStopping(monitor="train_loss", mode="min", patience=1000)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    trainer = pl.Trainer(
        max_epochs=N_EPOCHS,
        accelerator="gpu",
        devices=[0],
        accumulate_grad_batches=1,
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback, stopping_callback],
        logger=logger
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
