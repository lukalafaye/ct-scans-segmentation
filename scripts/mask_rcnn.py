import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "scripts")))

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from utils import load_dataset
import pandas as pd
from tqdm import tqdm

import torch.optim as optim

def collate_fn(batch):
    batch = [b for b in batch if b is not None]  # Filter out None entries
    return tuple(zip(*batch))

class MaskRCNNDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __getitem__(self, idx):
        image = self.images[idx]  # Shape: (256, 256)
        mask = self.masks[idx]    # Shape: (256, 256) or (N, 256, 256)

        # Convert grayscale image to 3-channel format (Mask R-CNN expects 3 channels)
        image = np.expand_dims(image, axis=0)  # (1, 256, 256)
        image = np.repeat(image, 3, axis=0)  # Convert to (3, 256, 256)
        image = torch.tensor(image, dtype=torch.float32) / 255.0  # Normalize

        # Convert masks to binary format
        obj_ids = np.unique(mask)[1:]  # Skip background (0)
        
        # If no objects are present, return None
        if len(obj_ids) == 0:
            return None

        masks = mask == obj_ids[:, None, None]  # Convert to shape (N, 256, 256)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        # Compute bounding boxes
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin, ymin, xmax, ymax = min(pos[1]), min(pos[0]), max(pos[1]), max(pos[0])
            if xmin == xmax:
                xmax += 1
            if ymin == ymax:
                ymax += 1
            boxes.append([xmin, ymin, xmax, ymax])

        # Ensure at least one bounding box exists
        if len(boxes) == 0:
            return None

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(obj_ids, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "masks": masks}

        return image, target

    def __len__(self):
        return len(self.images)

data_dir = "data/"
output_path = "data/submissions/mask_rcnn.csv"

y_train = pd.read_csv(f"{data_dir}\y_train.csv", index_col=0).T
y_train = y_train[:800] # only annotated images
y_train = y_train.values.reshape(800, 256, 256)
x_train = load_dataset(f"{data_dir}/train-images")
x_train = x_train[:800]
x_test = load_dataset(f"{data_dir}/test-images")

# Convert numpy arrays to dataset
train_dataset = MaskRCNNDataset(x_train, y_train)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# Load pretrained Mask R-CNN
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

# Modify the classifier to match your number of classes
num_classes = 55
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Modify the mask predictor
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1
for epoch in tqdm(range(num_epochs)):
    model.train()
    epoch_loss = 0
    
    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)  # Compute losses
        loss = sum(loss for loss in loss_dict.values())

        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

# labels.T.to_csv(output_csv_path)