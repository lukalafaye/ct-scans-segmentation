import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset
import os
from PIL import Image
import torchvision.transforms as transforms
import json
from utils import one_hot_mask, visualize_probabilistic_mask, visualize_image, visualize_probabilistic_mask_and_image
from utils import add_background_to_proba_mask
transform = transforms.ToTensor()

train_image_dir = "raidium_data/train-images"
test_image_dir = "raidium_data/test-images"
labels_csv_path = "raidium_data/y_train.csv"
df = pd.read_csv(labels_csv_path, index_col=0).T
image_filenames = list(df.index)

num_classes = 55

## Annotated labels
with open("raidium_data/annotated_labels.json") as file:
    annotated_labels = json.load(file)

annotated_idx = [i for i in range(len(annotated_labels)) if annotated_labels[i]]

## y train
masks = []
valid_filenames = []
for i in annotated_idx:
    valid_filenames.append(image_filenames[i])
    mask = (np.array(df.iloc[i]))
    mask = torch.tensor(mask, dtype=torch.uint8)
    mask = mask.reshape(256,256)
    mask_new = one_hot_mask(mask, num_classes).float()
    labels_possible = [0] + annotated_labels[i]
    zero_pixels = (mask == 0)
    for c in labels_possible:
        mask_new[c,zero_pixels] = 1.0 /len(labels_possible)
    masks.append(mask_new)
masks = torch.stack(masks)

## x train
images = []
for file in valid_filenames:
    image_path = os.path.join(train_image_dir, file)
    image = Image.open(image_path)
    image = transform(image)
    images.append(image)
images = torch.stack(images)

# Back to y train : modify masks based on black squares in image
masks_new = torch.stack([add_background_to_proba_mask(images[i], masks[i]) for i in range(masks.shape[0])])

# Create TensorDataset
dataset = TensorDataset(images, masks_new)

# Save dataset
torch.save(dataset, "raidium_data_tensor/train_dataset_probabilistic.pt")