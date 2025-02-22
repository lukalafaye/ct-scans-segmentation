import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from torchvision.datasets import VOCSegmentation
import segmentation_models_pytorch as smp
import numpy as np
import matplotlib.pyplot as plt

def visualize(image_tensor, mask_tensor, pred_tensor=None):
    """
    Utility function to visualize the image, ground truth mask,
    and optionally the predicted mask side by side.
    """
    # Convert tensors to numpy arrays
    image = image_tensor.permute(1, 2, 0).cpu().numpy()
    mask = mask_tensor.squeeze().cpu().numpy()
    
    # Clip or normalize if needed (some transforms might be required).
    image = np.clip(image, 0, 1)
    
    plt.figure(figsize=(12, 4))
    
    # Show original image
    plt.subplot(1, 3 if pred_tensor is not None else 2, 1)
    plt.title("Image")
    plt.imshow(image)
    
    # Show ground truth mask
    plt.subplot(1, 3 if pred_tensor is not None else 2, 2)
    plt.title("Ground Truth Mask")
    plt.imshow(mask, cmap='gray')
    
    # If prediction is given, show predicted mask
    if pred_tensor is not None:
        pred = pred_tensor.squeeze().cpu().numpy()
        plt.subplot(1, 3, 3)
        plt.title("Predicted Mask")
        plt.imshow(pred, cmap='gray')
    
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------
# 1. Hyperparameters & Setup
# -------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 5          
batch_size = 128         
lr = 1e-4
n_classes = 21          # Pascal VOC has 21 classes (20 + background)

# -------------------------------------------------------
# 2. Transforms and Dataset
# -------------------------------------------------------
# Use nearest interpolation for masks to avoid fractional class indices
input_size = (128, 128)

image_transform = T.Compose([
    T.Resize(input_size, interpolation=InterpolationMode.BILINEAR), # bilinear is okay for images
    T.ToTensor(),
])

target_transform = T.Compose([
    T.Resize(input_size, interpolation=InterpolationMode.NEAREST), # nearest for masks
    T.PILToTensor(),  # Keeps integer labels
])

root_dir = "data/voc/"
full_dataset = VOCSegmentation(
    root=root_dir,
    year="2012",
    image_set='train',
    download=True,
    transform=image_transform,
    target_transform=target_transform
)

print("Total samples in the dataset:", len(full_dataset))

# Dynamically determine train/val split
train_ratio = 0.8
train_size = int(train_ratio * len(full_dataset))
val_size = len(full_dataset) - train_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

# -------------------------------------------------------
# 3. Model (U-Net) from segmentation_models_pytorch
# -------------------------------------------------------
model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights="imagenet",
    in_channels=3,
    classes=n_classes
).to(device)

# Use ignore_index=255 to skip void pixels
criterion = nn.CrossEntropyLoss(ignore_index=255)
optimizer = optim.Adam(model.parameters(), lr=lr)

# -------------------------------------------------------
# 4. Training and Validation Loop
# -------------------------------------------------------
for epoch in range(num_epochs):
    print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
    
    # ---- Train ----
    model.train()
    running_loss = 0.0
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.squeeze(1).long().to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(train_loader)
    print(f"Training Loss: {epoch_loss:.4f}")
    
    # ---- Validation ----
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.squeeze(1).long().to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    print(f"Validation Loss: {val_loss:.4f}")

# -------------------------------------------------------
# 5. Visualize Some Predictions
# -------------------------------------------------------
model.eval()
examples_to_show = 2
with torch.no_grad():
    for i, (images, masks) in enumerate(val_loader):
        if i >= examples_to_show:
            break
        
        images = images.to(device)
        masks = masks.squeeze(1).to(device)
        outputs = model(images)  # shape [N, n_classes, H, W]
        
        # Take argmax along the channel dimension to get predicted class
        preds = torch.argmax(outputs, dim=1)
        
        # Show first sample in the batch
        visualize(
            image_tensor=images[0].cpu(),
            mask_tensor=masks[0].cpu(),
            pred_tensor=preds[0].cpu()
        )