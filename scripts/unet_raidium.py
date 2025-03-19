import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import TrainDataset, TestDataset
import segmentation_models_pytorch as smp
from tqdm import tqdm
from utils import dice_loss_no_bg, cross_entropy_no_bg, one_hot_mask_to_2d, dice_loss_one_hot
from utils import visualize_image, visualize_pred, save_mask_comparison, save_subplot_image
from utils import save_some_pred_vs_mask, predict_on_all_test
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


if __name__ == "__main__":

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(DEVICE)

    batch_size = 4
    num_epochs = 10
    num_classes = 55

    train_dataset_path = "raidium_data_tensor/train_dataset_probabilistic.pt"
    x_test_path = "raidium_data_tensor/x_test.pt"
    train_dataset = torch.load(train_dataset_path, weights_only=False)
    test_dataset = TestDataset(x_test_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=num_classes
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_losses = []
    val_losses = []

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            images = images.to(DEVICE)            # [B, 3, H, W]
            masks = masks.to(DEVICE)              

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)               
            loss = dice_loss_one_hot(masks, outputs, num_classes, eps=1e-7)

            # Backward
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch+1} train loss : {epoch_loss}")

    # 1 - Save visu of predictions vs. masks comparison on the train set
    # folder_name = "10_epochs_proba_detect"
    # save_some_pred_vs_mask(train_dataset, train_loader, model, batch_size, folder_name)

    # 2 - Predict on test set
    # predictions = predict_on_all_test(test_loader, model)

    # 3 - Save visualization of predictions on test
    # for i, image in enumerate(test_dataset):
    #     filename = f"test_preds_unet_dice_no_bg/{i}"
    #     save_subplot_image(image, (predictions[i].cpu()).reshape(256,256), filename)

    # 4 - Save predicted masks on test as csv
    # predictions = predictions.reshape(500, -1)
    # predictions = predictions.T
    # df = pd.DataFrame(predictions.cpu())
    # columns = [f"{i}.png" for i in range(500)]
    # index = [f"Pixel {i}" for i in range(256**2)]
    # df.columns = columns
    # df.index = index
    # df.to_csv("predictions_unet_basic_with_dice_no_bg.csv")




