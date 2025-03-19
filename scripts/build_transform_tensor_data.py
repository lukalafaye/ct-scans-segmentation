import pandas as pd
import numpy as np
import torch
import os
from PIL import Image
import torchvision.transforms as transforms

transform = transforms.ToTensor()

train_image_dir = "raidium_data/train-images"
test_image_dir = "raidium_data/test-images"
labels_csv_path = "raidium_data/y_train.csv"
df = pd.read_csv(labels_csv_path, index_col=0).T
image_filenames = list(df.index)


## y train
masks = []
valid_filenames = []
for i in range(df.shape[0]):
    mask = (np.array(df.iloc[i]))
    mask = torch.tensor(mask, dtype=torch.uint8)
    if mask.sum().item() != 0:
        mask = mask.reshape(256,256)
        masks.append(mask)
        valid_filenames.append(image_filenames[i])
masks = torch.stack(masks)
torch.save(masks, "raidium_data_tensor/y_train.pt")



## x train
images = []
for file in valid_filenames:
        image_path = os.path.join(train_image_dir, file)
        image = Image.open(image_path)
        image = transform(image)
        images.append(image)
images = torch.stack(images)
torch.save(images, "raidium_data_tensor/x_train.pt")



## x test

test_images = torch.zeros((500, 1, 256,256), dtype=torch.float64)
image_filenames = os.listdir(test_image_dir)
for i in range(500):
    image_path = os.path.join(test_image_dir, image_filenames[i])
    image = Image.open(image_path)
    image = transform(image)
    test_images[i] = image
torch.save(test_images, "raidium_data_tensor/x_test.pt")