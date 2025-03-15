import os
import json
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
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ----------------------------
# Step 0: Class Weights (Example)
# ----------------------------
frequencies = np.array([
    926, 673, 115092, 208579, 201305, 283963, 4723, 4422, 247582, 19671, 
    17542, 320273, 7937, 6131, 7501, 65158, 55008, 63491, 68319, 11931, 
    14037, 330252, 90404, 76409, 1977, 1859, 4612, 5045, 9206, 6423, 
    87612, 86498, 28015, 47374, 55493, 550923, 781124, 925175, 20693, 
    5564, 36685, 53261, 50937, 76503, 17677, 14275, 315423, 72506, 
    111266, 13885, 28022, 51741, 110911, 159006
])
median_freq = np.median(frequencies)
weights_non_background = median_freq / frequencies
weights = np.concatenate(([2.0], weights_non_background))
class_weights = torch.tensor(weights, dtype=torch.float32)

# Optional: give names to classes
class_names = {0: "Background"}
for i in range(1, 55):
    class_names[i] = f"Organ {i}"

# ----------------------------
# Step 1: Config/Paths
# ----------------------------
DATA_FOLDER = "../data"
TRAIN_IMAGES_DIR = os.path.join(DATA_FOLDER, "train-images")
Y_TRAIN_CSV = os.path.join(DATA_FOLDER, "y_train.csv")
VAL_PREDICTIONS_DIR = os.path.join(DATA_FOLDER, "unet_partial_predictions_b7_randomval")
DEBUGGING_DIR = os.path.join(DATA_FOLDER, "debugging_b7_randomval")
os.makedirs(VAL_PREDICTIONS_DIR, exist_ok=True)
os.makedirs(DEBUGGING_DIR, exist_ok=True)

annotated_labels = {}
annotated_labels_path = os.path.join(DATA_FOLDER, "annotated_labels.json")
if os.path.exists(annotated_labels_path):
    with open(annotated_labels_path, "r") as f:
        annotated_labels = json.load(f)

IMG_SIZE = 256
NUM_CLASSES = 55
BATCH_SIZE = 4
N_EPOCHS = 5000
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.use_deterministic_algorithms(True, warn_only=True)

# ----------------------------
# Step 2: Load & Filter CSV
# ----------------------------
df = pd.read_csv(Y_TRAIN_CSV, index_col=0).T
# Remove images that are entirely background (all zeros)
mask = ~(df.values == 0).all(axis=1)
df_filtered = df[mask]

print("Total filtered images:", len(df_filtered))
leftover_bg = (df_filtered.values == 0).all(axis=1).sum()
print("Still all-background after filtering:", leftover_bg)

# ----------------------------
# Step 3: Albumentations
# ----------------------------
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    ToTensorV2()
])
val_transform = A.Compose([ToTensorV2()])

# ----------------------------
# Step 4: Dataset
# ----------------------------
class SegmentationDataset(Dataset):
    def __init__(self, df, transform=None, annotated_labels=None):
        self.df = df
        self.filenames = list(df.index)
        self.transform = transform
        self.annotated_labels = annotated_labels

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image_path = os.path.join(TRAIN_IMAGES_DIR, filename)

        # Load grayscale image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Warning: Image not found at {image_path}")
            image = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        # Reshape mask from CSV
        mask_flat = self.df.loc[filename].values.astype(np.uint8)
        mask_2d = mask_flat.reshape(IMG_SIZE, IMG_SIZE)

        # Apply optional transforms
        if self.transform:
            aug = self.transform(image=image, mask=mask_2d)
            image_t = aug['image'].float() / 255.0
            mask_t  = aug['mask'].long()
        else:
            image_t = torch.from_numpy(image).float() / 255.0
            mask_t  = torch.from_numpy(mask_2d).long()

        # Remove background from JSON if present
        img_id = int(os.path.splitext(filename)[0])
        json_cls = set()
        if self.annotated_labels:
            json_cls = set(self.annotated_labels[img_id])
            if 0 in json_cls:
                json_cls.remove(0)

        return {
            "image": image_t,         # (1,H,W)
            "mask": mask_t,           # (H,W)
            "filename": filename,
            "json_classes": json_cls
        }

def custom_collate_fn(batch):
    images = [b["image"] for b in batch]
    masks  = [b["mask"]  for b in batch]
    filenames = [b["filename"] for b in batch]
    json_classes = [b["json_classes"] for b in batch]

    images = torch.stack(images, dim=0)  # (B,1,H,W)
    masks  = torch.stack(masks,  dim=0) # (B,H,W)

    return {
        "image": images,
        "mask": masks,
        "filename": filenames,
        "json_classes": json_classes
    }

# ----------------------------
# Step 5 (Modified): Train/Val Split
#  - Use the entire df_filtered as "train set"
#  - Randomly pick 30 images as "val set" for discrete dice monitoring
# ----------------------------
df_val = df_filtered.sample(n=30, random_state=SEED)
df_train = df_filtered.drop(df_val.index)

print(f"Train set size: {len(df_train)}")
print(f"Val set size:   {len(df_val)}")

train_dataset = SegmentationDataset(df_train, transform=train_transform, annotated_labels=annotated_labels)
val_dataset   = SegmentationDataset(df_val,   transform=val_transform,   annotated_labels=annotated_labels)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    collate_fn=custom_collate_fn
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=custom_collate_fn
)

# ----------------------------
# Step 6: UNet Model
# ----------------------------
base_model = smp.Unet(
    encoder_name="efficientnet-b7",
    encoder_weights="imagenet",
    in_channels=1,
    classes=NUM_CLASSES
)

# ----------------------------
# Step 7: Partial/Uncertain Losses
# ----------------------------
class PartialCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def build_valid_mask(self, mask, json_classes_batch):
        B, H, W = mask.shape
        device = mask.device
        valid_mask = torch.zeros((B, NUM_CLASSES, H, W), dtype=torch.bool, device=device)

        for b in range(B):
            mask_b = mask[b]
            json_cls = json_classes_batch[b]

            # definite organ classes in mask (excluding background=0)
            mask_classes = torch.unique(mask_b[mask_b != 0]).cpu().tolist()
            mask_cls_set = set(mask_classes)

            # "missing" classes that might appear in background
            missing = json_cls - mask_cls_set

            # 1) definite organ pixels => set valid_mask for that organ
            for c_val in mask_classes:
                idx = (mask_b == c_val)
                valid_mask[b, c_val, idx] = True

            # 2) background pixels => valid for class 0 + any missing classes
            idx_bg = (mask_b == 0)
            valid_mask[b, 0, idx_bg] = True
            for c_miss in missing:
                if 1 <= c_miss < NUM_CLASSES:
                    valid_mask[b, c_miss, idx_bg] = True

        return valid_mask

    def forward(self, logits, mask, json_classes_batch):
        B, C, H, W = logits.shape
        log_softmax = torch.log_softmax(logits, dim=1)

        with torch.no_grad():
            valid_mask = self.build_valid_mask(mask, json_classes_batch)

        exp_vals = torch.exp(log_softmax)
        exp_vals_valid = exp_vals * valid_mask.float()
        sum_of_probs = exp_vals_valid.sum(dim=1)

        eps = 1e-15
        loss_map = -torch.log(torch.clamp(sum_of_probs, min=eps))

        valid_pixel_mask = valid_mask.any(dim=1)
        valid_pixels = valid_pixel_mask.sum().item()
        if valid_pixels == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        return (loss_map * valid_pixel_mask.float()).sum() / valid_pixels

class PartialDiceLoss(nn.Module):
    def __init__(self, smooth=1e-7):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, mask, json_classes_batch):
        B, C, H, W = logits.shape
        probs = torch.softmax(logits, dim=1)
        probs_flat = probs.view(B, C, -1)
        mask_flat  = mask.view(B, -1)

        dice_vals = []
        for b in range(B):
            mask_b = mask_flat[b]
            probs_b = probs_flat[b]
            organ_vals = torch.unique(mask_b[mask_b != 0])
            if len(organ_vals) == 0:
                continue

            per_organ_dice = []
            for c_org in organ_vals:
                c_val = c_org.item()
                pred_c = probs_b[c_val, :]
                gt_c   = (mask_b == c_val).float()

                inter = (pred_c * gt_c).sum()
                denom = pred_c.sum() + gt_c.sum() + self.smooth
                dice_c = 2.0 * inter / denom
                per_organ_dice.append(dice_c)

            if per_organ_dice:
                mean_dice = torch.stack(per_organ_dice).mean()
                dice_vals.append(mean_dice)

        if len(dice_vals) == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        return 1.0 - torch.stack(dice_vals).mean()

class PartialCombinedLoss(nn.Module):
    def __init__(self, ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.ce_loss = PartialCrossEntropyLoss()
        self.dice_loss = PartialDiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, mask, json_cls_batch):
        loss_ce = self.ce_loss(logits, mask, json_cls_batch)
        loss_dice = self.dice_loss(logits, mask, json_cls_batch)
        combined = self.ce_weight * loss_ce + self.dice_weight * loss_dice
        return combined, loss_ce, loss_dice

# ----------------------------
# Step 8: Discrete Dice & Enhanced Callback
# ----------------------------
def partial_discrete_dice(logits, mask, json_cls_batch):
    preds = torch.argmax(logits, dim=1)
    B, H, W = preds.shape

    dice_list = []
    for b in range(B):
        mask_b = mask[b].view(-1)
        pred_b = preds[b].view(-1)
        organ_vals = torch.unique(mask_b[mask_b != 0])
        if len(organ_vals) == 0:
            continue

        dices = []
        for c_org in organ_vals:
            c_val = c_org.item()
            pred_mask = (pred_b == c_val).float()
            gt_mask   = (mask_b == c_val).float()
            inter = (pred_mask * gt_mask).sum()
            denom = pred_mask.sum() + gt_mask.sum()
            if denom > 0:
                dices.append(2.0 * inter / denom)
        if len(dices) > 0:
            dice_list.append(torch.stack(dices).mean())

    if len(dice_list) == 0:
        return torch.tensor(1.0, device=logits.device)
    return torch.stack(dice_list).mean()

class EnhancedValPredictionCallback(Callback):
    def __init__(self, val_dataloader, output_dir, save_every=3, img_size=IMG_SIZE, class_names=None):
        super().__init__()
        self.val_dataloader = val_dataloader
        self.output_dir = output_dir
        self.save_every = save_every
        self.img_size = img_size
        self.class_names = class_names if class_names else {}
        os.makedirs(self.output_dir, exist_ok=True)

        # Create a colormap with background=white
        cmap_colors = plt.cm.jet(np.linspace(0, 1, NUM_CLASSES))
        cmap_colors[0] = [1,1,1,1]
        self.cmap = ListedColormap(cmap_colors)

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.save_every != 0:
            return

        epoch_dir = os.path.join(self.output_dir, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        
        device = pl_module.device
        pl_module.model.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
                images = batch['image'].to(device)
                masks_np = batch['mask'].cpu().numpy()
                filenames = batch['filename']
                json_classes_list = batch['json_classes']

                logits = pl_module(images)
                preds_np = torch.argmax(logits, dim=1).cpu().numpy()

                for i in range(images.size(0)):
                    img_t = images[i].cpu().squeeze(0).numpy()
                    mask_gt = masks_np[i]
                    mask_pred = preds_np[i]
                    fname = filenames[i]
                    json_cls = json_classes_list[i]

                    mask_gt_unique = np.unique(mask_gt)
                    gt_classes_list = [c for c in mask_gt_unique if c != 0]
                    mask_pred_unique = np.unique(mask_pred)
                    pred_classes_list = [c for c in mask_pred_unique if c != 0]

                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    axes[0].imshow(img_t, cmap='gray', vmin=0, vmax=1)
                    axes[0].set_title('Original Image')
                    axes[0].axis('off')

                    axes[1].imshow(mask_gt, cmap=self.cmap, vmin=0, vmax=NUM_CLASSES-1)
                    axes[1].set_title('Ground Truth')
                    axes[1].axis('off')

                    axes[2].imshow(mask_pred, cmap=self.cmap, vmin=0, vmax=NUM_CLASSES-1)
                    axes[2].set_title('Prediction')
                    axes[2].axis('off')

                    info_str = (
                        f"JSON classes: {sorted(list(json_cls))}\n"
                        f"Mask classes: {sorted(gt_classes_list)}\n"
                        f"Pred classes: {sorted(pred_classes_list)}\n"
                    )
                    plt.suptitle(f"File: {fname} | Epoch {epoch}")
                    plt.figtext(0.02, 0.01, info_str, fontsize=9)
                    plt.tight_layout()

                    out_name = f"val_{batch_idx}_{i}_{os.path.basename(fname)}.png"
                    save_path = os.path.join(epoch_dir, out_name)
                    plt.savefig(save_path, dpi=150)
                    plt.close(fig)
        pl_module.model.train()

# ----------------------------
# Step 9: Lightning Module
# ----------------------------
class UNetLightning(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3, ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.model = model
        self.loss_fn = PartialCombinedLoss(ce_weight=ce_weight, dice_weight=dice_weight)
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['mask']
        j = batch['json_classes']

        logits = self(x)
        loss_combined, loss_ce, loss_dice = self.loss_fn(logits, y, j)

        self.log("train_combined_loss", loss_combined, prog_bar=True, on_epoch=True)
        self.log("train_ce_loss",       loss_ce,       on_epoch=True)
        self.log("train_dice_loss",     loss_dice,     on_epoch=True)

        return loss_combined

    def validation_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['mask']
        j = batch['json_classes']

        logits = self(x)
        loss_combined, loss_ce, loss_dice = self.loss_fn(logits, y, j)
        disc_dice_val = partial_discrete_dice(logits, y, j)

        self.log("val_combined_loss", loss_combined, on_epoch=True, prog_bar=True)
        self.log("val_ce_loss",       loss_ce,       on_epoch=True)
        self.log("val_dice_loss",     loss_dice,     on_epoch=True)
        self.log("val_discrete_dice", disc_dice_val, on_epoch=True, prog_bar=True)

        return {
            "val_combined_loss": loss_combined,
            "val_ce_loss":       loss_ce,
            "val_dice_loss":     loss_dice,
            "val_discrete_dice": disc_dice_val
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=7, min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_combined_loss"
        }

# ----------------------------
# (Optional) Visualization
# ----------------------------
def visualize_dataset_samples(dataset, num_samples=5, save_dir=DEBUGGING_DIR):
    os.makedirs(save_dir, exist_ok=True)
    cmap_colors = plt.cm.jet(np.linspace(0, 1, NUM_CLASSES))
    cmap_colors[0] = [1, 1, 1, 1]
    cmap = ListedColormap(cmap_colors)

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        image = sample['image'].squeeze().numpy()
        mask  = sample['mask'].numpy()
        fname = sample['filename']
        json_cls = sample['json_classes']

        unique_classes, counts = np.unique(mask, return_counts=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Image')
        axes[0].axis('off')

        axes[1].imshow(mask, cmap=cmap, vmin=0, vmax=NUM_CLASSES-1)
        axes[1].set_title('Mask')
        axes[1].axis('off')

        plt.suptitle(f"File: {fname}\nJSON classes: {sorted(list(json_cls))}")

        info_str = ""
        for c_val, c_count in zip(unique_classes, counts):
            if c_val != 0:
                info_str += f"Class {c_val} => {c_count} pixels\n"
        plt.figtext(0.02, 0.02, info_str, fontsize=9)
        save_path = os.path.join(save_dir, f"sample_{i}_{fname}")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    print(f"Saved {num_samples} sample visualizations to {save_dir}")

# ----------------------------
# Step 10: main
# ----------------------------
def main():
    logger = TensorBoardLogger("tb_logs", name="unet_partial_seg_b4_randomval")

    # (Optional) Visualize a few samples from the training dataset:
    # visualize_dataset_samples(train_dataset, num_samples=5)

    # Construct the Lightning Trainer:
    trainer = pl.Trainer(
        max_epochs=N_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=10,
        callbacks=[
            # Stop training if val_discrete_dice doesn't improve for 30 epochs
            EarlyStopping(monitor="val_discrete_dice", mode="max", patience=30),
            # Save top-5 checkpoints based on highest val_discrete_dice
            ModelCheckpoint(
                dirpath="models/",
                filename="b7partial-randval-unet-{epoch:02d}-{val_discrete_dice:.4f}",
                monitor="val_discrete_dice",
                save_top_k=5,
                mode="max"
            ),
            # Callback for saving validation predictions every 5 epochs
            EnhancedValPredictionCallback(
                val_loader,
                VAL_PREDICTIONS_DIR,
                save_every=5,
                class_names=class_names
            )
        ],
        logger=logger,
        gradient_clip_val=1.0
    )

    model_lightning = UNetLightning(
        model=base_model,
        learning_rate=3e-3,
        ce_weight=0.5,
        dice_weight=0.5
    )

    print("Starting training on entire dataset + random 30-image val set ...")
    trainer.fit(model_lightning, train_loader, val_loader)
    print("Training completed!")

if __name__ == "__main__":
    main()
