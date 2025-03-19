import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np



class SegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, rgb_labels, transform):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))
        self.rgb_labels = rgb_labels
        self.transform = transform

        assert len(self.image_filenames) == len(self.mask_filenames), "Mismatch in image and mask count !"

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):

        # img
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        img = Image.open(img_path)
        img = self.transform(img)
        pad_h = 512 - img.shape[1]
        pad_w = 512 - img.shape[2]
        img = F.pad(img, (0, pad_w, 0, pad_h))

        # mask
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])
        mask = Image.open(mask_path).convert('RGB')
        mask = np.array(mask)
        mask_new = np.zeros(mask.shape[:2])
        for i, color in enumerate(self.rgb_labels):
            condition = np.all(mask == color, axis=-1)
            mask_new[condition] = i
        mask = F.pad(torch.tensor(mask_new), (0, pad_w, 0, pad_h))
        mask = mask.to(dtype=torch.int64)
        return img, mask


def visualize_img(img):
    plt.imshow((img.permute(1,2,0)).numpy())
    plt.axis('off')
    plt.show()

def visualize_mask(mask, rgb_labels):
    mask_new = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    for i, color in enumerate(rgb_labels):
        condition = (mask == i)  # Generate a boolean mask
        mask_new[condition] = color  # Apply color
    plt.imshow(mask_new)
    plt.axis('off')
    plt.show()

def visualize_pred(pred, rgb_labels):
    pred = pred.cpu()
    mask_new = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    
    for i, color in enumerate(rgb_labels):
        condition = (pred == i)  # Generate a boolean mask
        mask_new[condition] = color  # Apply color
    plt.imshow(mask_new)
    plt.axis('off')
    plt.show()

def dice_loss(true, logits, num_classes, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    true_new = true.reshape(4, 1, 512, 512)
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true_new.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes, device=DEVICE)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)



if __name__ == "__main__":

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    transform = transforms.ToTensor()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(DEVICE)

    train_image_dir = 'voc_2012_segmentation_data/train_images'
    train_mask_dir = 'voc_2012_segmentation_data/train_labels'
    val_image_dir = 'voc_2012_segmentation_data/valid_images'
    val_mask_dir = 'voc_2012_segmentation_data/valid_labels'

    train_img_filenames = os.listdir(train_image_dir)
    train_mask_filenames = os.listdir(train_mask_dir)

    img = os.path.join(train_image_dir, train_img_filenames[0])
    mask = os.path.join(train_mask_dir, train_mask_filenames[0])

    batch_size = 4
    num_epochs = 20

    rgb_labels = [[0, 0, 0],
                    [224, 224, 192],
                    [128, 0, 0],
                    [192, 128, 128],
                    [0, 64, 128],
                    [192, 0, 0],
                    [64, 0, 128],
                    [128, 128, 0],
                    [128, 0, 128],
                    [0, 0, 128],
                    [192, 128, 0],
                    [128, 192, 0],
                    [64, 128, 128],
                    [192, 0, 128],
                    [64, 128, 0],
                    [128, 128, 128],
                    [0, 128, 0],
                    [64, 0, 0],
                    [0, 192, 0],
                    [0, 128, 128],
                    [0, 64, 0],
                    [128, 64, 0]]

    num_classes = len(rgb_labels)

    train_dataset = SegDataset(train_image_dir, train_mask_dir, rgb_labels, transform)
    val_dataset = SegDataset(val_image_dir, val_mask_dir, rgb_labels, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # ---------------------------------------------------------
    # 2. Define your model, loss, and optimizer
    # ---------------------------------------------------------
    # We'll use a ResNet-34 backbone as an example.

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # ---------------------------------------------------------
    # 3. Training Loop
    # ---------------------------------------------------------

    train_losses = []
    val_losses = []

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            images = images.to(DEVICE)            # [B, 3, H, W]
            masks = masks.long().to(DEVICE)              

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)               
            #loss = criterion(outputs, masks)
            loss = dice_loss(masks, outputs,num_classes, eps=1e-7)

            # Backward
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        with torch.no_grad():
            running_val_loss = 0.0
            for images, masks in val_loader:
                images = images.to(DEVICE)
                masks = masks.long().to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, masks)
                running_val_loss += loss.item()

            val_loss = running_val_loss / len(val_loader)
            val_losses.append(val_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train loss: {epoch_loss:.4f}, Val loss : {val_loss:.4f}")


    # Visualize some images / ground truth / predictions

    images, masks = next(iter(val_loader))
    images = images.to(DEVICE)
    masks = masks.long().to(DEVICE)
    logits = model(images)
    preds = torch.argmax(logits, dim=1)
    for i in range(batch_size):
        img, mask = val_dataset[i]
        pred = preds[i].reshape(512,512).cpu()
        visualize_img(img)
        visualize_mask(mask, rgb_labels)
        visualize_mask(pred, rgb_labels)

    # Plot the evolution of the val and train losses
    plt.figure()
    plt.plot(train_losses, label="train loss")
    plt.plot(val_losses, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()