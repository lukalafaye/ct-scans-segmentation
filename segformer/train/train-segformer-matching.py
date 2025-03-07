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
#train_df = train_df.iloc[:50]

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
        # Create one-hot masks for each unique label
        y = [y == x for x in np.unique(y)]
        # Separate background (first channel) and foreground channels
        y_ = y[1:]
        # Shuffle the foreground channels to avoid fixed ordering
        random.shuffle(y_)
        y = np.stack(y[:1] + y_)
        # Pad to MAX_ITEMS channels if necessary
        y = np.pad(y, [(0, MAX_ITEMS - y.shape[0]), (0, 0), (0, 0)])
        return y.astype(np.float32)
    
    def __len__(self):
        return len(self.paths)

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# Suppose train_df is your DataFrame containing image names as index and 
# the segmentation labels as values (flattened or not) as read from your CSV.
# We'll create a multi-label indicator matrix where each row corresponds 
# to an image and each column corresponds to one of the 55 classes.

# For each image, compute a binary vector indicating presence/absence of each class.
def compute_label_vector(row, max_items=55):
    # Get the unique labels present in the row (assumes background is 0)
    labels = np.unique(row.values)
    # Create a binary vector of length max_items
    vec = np.zeros(max_items, dtype=int)
    # For each label (assuming labels are in [0, max_items-1]),
    # mark as present (if label > 0, e.g., skipping background if desired)
    for lbl in labels:
        # Optionally skip background (if background is 0 and you don't want to stratify on it)
        if lbl != 0:
            vec[int(lbl)] = 1
    return vec

# Compute the multi-label indicator for each image
multi_labels = np.stack([compute_label_vector(train_df.loc[name]) for name in train_df.index])

# Now use MultilabelStratifiedKFold to create a split.
mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
train_indices, val_indices = next(mskf.split(np.zeros(len(train_df)), multi_labels))

# Create separate DataFrames for train and validation.
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
    shuffle=False  # typically no shuffling for validation
)

print("Train split contains: ", len(train_df_split.stack().unique()), " - Val split contains: ", len(val_df_split.stack().unique()))

# Dice Loss definition
class DiceLoss(torch.nn.Module):
    def __init__(self,):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-10):
        # Flatten tensors: both become vectors of shape (B * MAX_ITEMS * 256*256,)
        inputs, targets = inputs.contiguous().view(-1), targets.contiguous().view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice  # scalar

# Matching Loss definition modified to return a tuple (ce_loss, dice_loss)
class MatchingLoss(torch.nn.Module):
    def __init__(self,):
        super(MatchingLoss, self).__init__()
        # Cross entropy function expects raw logits (it applies softmax internally)
        self.loss_fn1 = torch.nn.functional.cross_entropy
        self.loss_fn2 = DiceLoss()
    
    def forward(self, inputs, targets):
        # inputs: (B, MAX_ITEMS, 256, 256) logits; targets: (B, MAX_ITEMS, 256, 256) one-hot masks
        # Reshape to flatten spatial dimensions: (B, MAX_ITEMS, 256*256)
        inputs = inputs.view(*inputs.size()[:2], -1)
        targets = targets.view(*targets.size()[:2], -1)

        # Remove the background channel (index 0)
        bg_inputs, bg_targets = inputs[:, :1], targets[:, :1]  # (B, 1, 256*256)
        inputs, targets = inputs[:, 1:], targets[:, 1:]           # (B, MAX_ITEMS-1, 256*256)

        # Compute best matching loss between foreground channels using L1 cost
        costs = torch.cdist(inputs, targets, p=1).detach().cpu().numpy()  # (B, MAX_ITEMS-1, MAX_ITEMS-1)
        inputs_arr, targets_arr = [], []
        for i in range(len(costs)):  # loop over batch (B)
            inputs_, targets_ = inputs[i], targets[i]  # each: (MAX_ITEMS-1, 256*256)
            l, k = linear_sum_assignment(costs[i])  # optimal matching indices for foreground channels
            inputs_arr.append(inputs_[l])   # reorder channels: (MAX_ITEMS-1, 256*256)
            targets_arr.append(targets_[k])  # reorder channels: (MAX_ITEMS-1, 256*256)
        inputs = torch.stack(inputs_arr)    # (B, MAX_ITEMS-1, 256*256)
        targets = torch.stack(targets_arr)   # (B, MAX_ITEMS-1, 256*256)

        # Add background channel back
        inputs = torch.cat([bg_inputs, inputs], 1)   # Final shape: (B, MAX_ITEMS, 256*256)
        targets = torch.cat([bg_targets, targets], 1)  # Final shape: (B, MAX_ITEMS, 256*256)

        # For cross entropy, convert one-hot targets to indices (each pixel: integer in [0, MAX_ITEMS-1])
        target_indices = torch.argmax(targets, dim=1)  # shape: (B, 256*256)

        ce_loss = self.loss_fn1(inputs, target_indices)
        dice_loss = self.loss_fn2(torch.softmax(inputs, dim=1), targets)
        return ce_loss, dice_loss

# LightningModule definition with TensorBoard logging and epoch-level logging using the loss_fn outputs.
class MyModel(pl.LightningModule):
    def __init__(self, in_channels=1):
        super().__init__()

        # Load the config of the pretrained nvidia/mit-b* model
        config = SegformerConfig.from_pretrained(MODEL_NAME)
        config.num_channels = in_channels  # 1 for grayscale
        config.id2label = {i: i for i in range(MAX_ITEMS)}
        config.label2id = {i: i for i in range(MAX_ITEMS)}
        self.config = config
        # Create a randomly initialized model from the config
        self.backbone = SegformerForSemanticSegmentation(config)
        
        self.loss_fn = MatchingLoss()
        self.mean_tracker = MeanMetric()

    def forward(self, img):
        # img shape: (B, 1, 256, 256)
        img = img / 255
        out = self.backbone(pixel_values=img)[0]  # (B, MAX_ITEMS, 64, 64) downsampled (example)
        out = torch.nn.functional.interpolate(out, mode='bilinear', scale_factor=4)  # upsample to (B, MAX_ITEMS, 256, 256)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch  # x: (B, 1, 256, 256); y: (B, MAX_ITEMS, 256, 256)
        y_hat = self(x)  # y_hat: (B, MAX_ITEMS, 256, 256)
        ce_loss, dice_loss = self.loss_fn(y_hat, y)  # each is a scalar loss
        loss = ce_loss + dice_loss
        
        # Compute dice score from dice_loss
        dice_score = 1 - dice_loss
        
        # Log metrics (on_epoch=True aggregates over the epoch)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_ce_loss", ce_loss, prog_bar=True, on_epoch=True)
        self.log("train_dice_score", dice_score, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        ce_loss, dice_loss = self.loss_fn(y_hat, y)
        loss = ce_loss + dice_loss
        
        # Compute dice score from dice_loss
        dice_score = 1 - dice_loss
        
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_ce_loss", ce_loss, prog_bar=True, on_epoch=True)
        self.log("val_dice_score", dice_score, prog_bar=True, on_epoch=True)
        return {"val_loss": loss, "val_ce_loss": ce_loss, "val_dice_score": dice_score}

    def on_train_epoch_end(self):
        # Reset the mean tracker at the end of each epoch
        self.mean_tracker.reset()

    def on_validation_epoch_end(self):
        # No extra aggregation is required here since we log on_epoch in each step.
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
logger = TensorBoardLogger("tb_logs", name="segform")

# Device selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN = True

if TRAIN:
    model = MyModel()
    """
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="models/",
        filename="model",
        save_top_k=1,
        monitor="train_loss",
        mode="min",
        every_n_epochs=1,
        save_weights_only=True,
        verbose=True
    )
    """

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="models/",
        filename="best_model_matching",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        every_n_epochs=5,  # check and save every 5 epochs
        save_weights_only=True,
        verbose=True
    )

    stopping_callback = EarlyStopping(monitor="val_loss", mode="min", patience=50)
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
    # Provide both train and validation dataloaders
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


