import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
color_map = [[0, 0, 0]] + [[int(r), int(g), int(b)] for r, g, b in plt.cm.viridis(np.linspace(0, 1, 54))[:, :3] * 255]
probabilistic_color_map = color_map + [[255, 255, 255]] # add white for uncertain pixels
def visualize_image(img_tensor):
    plt.imshow(img_tensor.squeeze(0).numpy(), cmap='gray')
    plt.axis('off')
    plt.show()

def visualize_mask(mask_tensor, color_map = color_map):
    mask_new = np.zeros((mask_tensor.shape[0], mask_tensor.shape[1], 3), dtype=np.uint8)
    
    for i, color in enumerate(color_map):
        condition = (mask_tensor == i) 
        mask_new[condition] = color
    plt.imshow(mask_new)
    plt.axis('off')
    plt.show()

def visualize_probabilistic_mask(mask, color_map = probabilistic_color_map):
    """
    mask : tensor of shape [C (num_classes), H, W] with float values between 0 and 1
    values 0 and 1 are considered as certain and correspond to a specific color
    values 0 < v < 1 are considered as uncertain and treated as an extra class. 
    They appear as white (last color of the color map)
    """
    mask_2d = np.full((mask.shape[1], mask.shape[2]), fill_value = -1, dtype = np.int32)
    for c in range(mask.shape[0]):
        condition = (mask[c].cpu().numpy() == 1.0)
        mask_2d[condition] = c
    uncertain_condition = (mask_2d == -1)
    mask_2d[uncertain_condition] = mask.shape[0]
    mask_visu = np.zeros((mask.shape[1], mask.shape[2], 3), dtype = np.uint8)
    for i, color in enumerate(color_map):
        condition = (mask_2d == i)
        mask_visu[condition] = color
    plt.imshow(mask_visu)
    plt.axis('off')
    plt.show()

def one_hot_mask_to_2d(mask):
    mask_2d = np.full((mask.shape[1], mask.shape[2]), fill_value=-1, dtype=np.int32)
    for c in range(mask.shape[0]):
        condition = (mask[c].cpu().numpy() == 1.0)
        mask_2d[condition] = c
    return mask_2d


def visualize_probabilistic_mask_and_image(img_tensor, mask, alpha=0.5, color_map=probabilistic_color_map):
    """
    Overlays a probabilistic segmentation mask onto an image.
    
    Parameters:
    img_tensor : torch.Tensor of shape [1, H, W] (grayscale image)
    mask : torch.Tensor of shape [C (num_classes), H, W] with float values between 0 and 1
    alpha : float, transparency level for overlaying the mask
    color_map : list of RGB colors, including an extra color for uncertainty
    """
    img = img_tensor.squeeze(0).numpy()
    
    # Convert probabilistic mask to discrete mask representation
    mask_2d = np.full((mask.shape[1], mask.shape[2]), fill_value=-1, dtype=np.int32)
    for c in range(mask.shape[0]):
        condition = (mask[c].cpu().numpy() == 1.0)
        mask_2d[condition] = c
    
    # Assign uncertain values to the extra class (last color in the map)
    uncertain_condition = (mask_2d == -1)
    mask_2d[uncertain_condition] = mask.shape[0]
    
    # Create RGB mask
    mask_visu = np.zeros((mask.shape[1], mask.shape[2], 3), dtype=np.uint8)
    for i, color in enumerate(color_map):
        condition = (mask_2d == i)
        mask_visu[condition] = color
    
    # Normalize grayscale image to [0, 1] for display
    img_normalized = (img - img.min()) / (img.max() - img.min())
    img_rgb = np.stack([img_normalized] * 3, axis=-1)
    
    # Overlay mask with transparency
    overlay = (1 - alpha) * img_rgb + (alpha * mask_visu / 255.0)
    
    plt.imshow(overlay)
    plt.axis('off')
    plt.show()
    

def visualize_mask_and_image(img_tensor, mask_tensor, alpha=0.5, color_map = color_map):
    img = img_tensor.squeeze(0).numpy()
    mask_new = np.zeros((mask_tensor.shape[0], mask_tensor.shape[1], 3), dtype=np.uint8)

    for i, color in enumerate(color_map):
        condition = (mask_tensor == i)
        mask_new[condition] = color

    # Normalize image to [0, 1] for proper display
    img_normalized = (img - img.min()) / (img.max() - img.min())

    # Convert grayscale to RGB
    img_rgb = np.stack([img_normalized] * 3, axis=-1)

    # Overlay mask with transparency
    overlay = (1 - alpha) * img_rgb + (alpha * mask_new / 255.0)

    plt.imshow(overlay)
    plt.axis('off')
    plt.show()

def visualize_pred(pred, color_map = color_map):
    pred = pred.cpu()
    mask_new = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for i, color in enumerate(color_map):
        condition = (pred == i)
        mask_new[condition] = color
    plt.imshow(mask_new)
    plt.axis('off')
    plt.show()

def save_subplot_image(img_tensor, mask_tensor, filename, alpha=0.5, color_map = color_map):
    img = img_tensor.squeeze(0).numpy()
    mask_new = np.zeros((mask_tensor.shape[0], mask_tensor.shape[1], 3), dtype=np.uint8)

    for i, color in enumerate(color_map):
        condition = (mask_tensor == i)
        mask_new[condition] = color

    # Normalize image to [0, 1] for proper display
    img_normalized = (img - img.min()) / (img.max() - img.min())

    # Convert grayscale to RGB
    img_rgb = np.stack([img_normalized] * 3, axis=-1)

    # Overlay mask with transparency
    overlay = (1 - alpha) * img_rgb + (alpha * mask_new / 255.0)
    
    # Create subplot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_rgb, cmap='gray')
    axes[0].axis('off')

    
    axes[1].imshow(mask_new)
    axes[1].axis('off')

    
    axes[2].imshow(overlay)
    axes[2].axis('off')

    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def save_mask_comparison(img_tensor, gt_mask_tensor, pred_mask_tensor, filename, alpha=0.5, color_map = color_map):
    img = img_tensor.squeeze(0).numpy()

    def create_colored_mask(mask_tensor):
        mask_rgb = np.zeros((mask_tensor.shape[0], mask_tensor.shape[1], 3), dtype=np.uint8)
        for i, color in enumerate(color_map):
            mask_rgb[mask_tensor == i] = color
        return mask_rgb

    # Process masks
    gt_mask_rgb = create_colored_mask(gt_mask_tensor)
    pred_mask_rgb = create_colored_mask(pred_mask_tensor)

    # Normalize image
    img_normalized = (img - img.min()) / (img.max() - img.min())
    img_rgb = np.stack([img_normalized] * 3, axis=-1)

    # Overlay masks
    gt_overlay = (1 - alpha) * img_rgb + (alpha * gt_mask_rgb / 255.0)
    pred_overlay = (1 - alpha) * img_rgb + (alpha * pred_mask_rgb / 255.0)

    # Create subplot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(img_rgb, cmap='gray')
    axes[0].axis('off')

    axes[1].imshow(gt_overlay)
    axes[1].axis('off')

    axes[2].imshow(pred_overlay)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def save_some_pred_vs_mask(train_dataset, train_loader, model, batch_size, folder_name):
    model.eval()
    images, masks = next(iter(train_loader))
    images, masks = next(iter(train_loader))
    images = images.to(DEVICE)
    masks = masks.to(DEVICE)
    logits = model(images)
    preds = torch.argmax(logits, dim=1)
    for i in range(batch_size):
        img, mask = train_dataset[i]
        pred = preds[i].cpu()
        filename = f"{folder_name}/{i}.png"
        save_mask_comparison(img, one_hot_mask_to_2d(mask), pred, filename)


def predict_on_all_test(test_loader, model):
    predictions = []
    for images in test_loader:
        images = images.to(DEVICE)
        logits = model(images)
        preds = torch.argmax(logits, dim=1)
        predictions.append(preds)
    return torch.cat(predictions, dim=0)

def dice_loss(masks, logits, num_classes, eps=1e-7):
    """
    masks : tensor of shape [B, H, W]
    logits : tensor of shape [B, C (num_classes), H, W]
    """
    masks_one_hot = torch.zeros((masks.shape[0], num_classes, masks.shape[1], masks.shape[2])).to(DEVICE)
    for c in range(num_classes):
        masks_one_hot[:,c,:,:] = (masks == c)
    probas = F.softmax(logits, dim=1)
    dice_coeffs = []
    for c in range(num_classes):
        y = masks_one_hot[:,c,:,:].flatten()
        z = probas[:,c,:,:].flatten()
        dice_coeffs.append((2*torch.dot(y,z)/(torch.sum(y) + torch.sum(z) + eps)))
    return 1 - torch.mean(torch.stack(dice_coeffs))

def one_hot_mask(mask, num_classes):
    """
    mask : tensor of shape [H,W] with integers
    returns : tensor of shape [C, H, W] with zero and ones
    """
    mask_one_hot = torch.zeros((num_classes, mask.shape[0], mask.shape[1]))
    for c in range(num_classes):
        mask_one_hot[c,:,:] = (mask == c)
    return mask_one_hot

def dice_loss_no_bg(masks, logits, num_classes, eps=1e-7):
    """
    masks : tensor of shape [B, H, W]
    logits : tensor of shape [B, C (num_classes), H, W]
    """
    masks_one_hot = torch.zeros((masks.shape[0], num_classes - 1, masks.shape[1], masks.shape[2])).to(DEVICE) # use num_classes - 1 classes this time
    for c in range(1,num_classes): # start at 1 to ignore background
        masks_one_hot[:,c - 1,:,:] = (masks == c)
    logits = logits[:,1:,:,:] # Remove the first slice along dimension 1 (background)
    probas = F.softmax(logits, dim=1)
    dice_coeffs = []
    for c in range(num_classes-1):
        y = masks_one_hot[:,c,:,:].flatten()
        z = probas[:,c,:,:].flatten()
        dice_coeffs.append((2*torch.dot(y,z)/(torch.sum(y) + torch.sum(z) + eps)))
    return 1 - torch.mean(torch.stack(dice_coeffs))

def dice_loss_one_hot(masks, logits, num_classes, eps=1e-7):
    """
    masks : tensor of shape [B,C,H,W]
    logits : tensor of shape [B,C,H,W]
    """
    probas = F.softmax(logits, dim=1)
    dice_coeffs = []
    for c in range(num_classes):
        y = masks[:,c,:,:].flatten()
        z = probas[:,c,:,:].flatten()
        dice_coeffs.append((2*torch.dot(y,z)/(torch.sum(y) + torch.sum(z) + eps)))
    return 1 - torch.mean(torch.stack(dice_coeffs))

def cross_entropy_no_bg(masks, logits, num_classes):
    """
    masks : tensor of shape [B, H, W]
    logits : tensor of shape [B, C (num_classes), H, W]
    """
    # Convert masks to one-hot format (excluding background)
    masks_one_hot = F.one_hot(masks, num_classes=num_classes).permute(0, 3, 1, 2)[:, 1:].to(DEVICE)  
    logits = logits[:, 1:]  # Exclude background class

    # Compute log-softmax to prevent numerical issues
    log_probs = F.log_softmax(logits, dim=1)

    # Compute cross-entropy loss (sum over classes first)
    loss = -(masks_one_hot * log_probs).sum(dim=1).mean()  # Fix: No extra stacking

    return loss


def dice_loss_old(true, logits, num_classes, eps=1e-7):
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
    true_new = true.reshape(true.shape[0], 1, true.shape[1], true.shape[2])
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


class TrainDataset(Dataset):
    def __init__(self, x_train_path, y_train_path):
        self.x_train = torch.load(x_train_path)
        self.y_train = torch.load(y_train_path)

    def __len__(self):
        return self.x_train.shape[0]

    def __getitem__(self, idx):
        return self.x_train[idx].float(), self.y_train[idx]
    
class TestDataset(Dataset):
    def __init__(self, x_test_path):
        self.x_test = torch.load(x_test_path)

    def __len__(self):
        return self.x_test.shape[0]

    def __getitem__(self, idx):
        return self.x_test[idx].float()
    
def neighborhood_has_colors(i_pixel, j_pixel, image, r=6, threshold = 0.08):
    H, W = image.shape[1:]
    i_min = max(0, i_pixel - r)
    i_max = min(H - 1, i_pixel + r)
    j_min = max(0, j_pixel - r)
    j_max = min(W - 1, j_pixel + r)
    image_neighborhood = image[0,i_min :i_max+1, j_min:j_max+1]
    return ((image_neighborhood > threshold).any()).item()
    


def add_background_to_proba_mask(image, mask, r=7, threshold = 0.1):
    """
    image : a [1,H,W] tensor
    mask : a [C,H,W] tensor with float values between 0 and 1
    Initially, the mask only contains uncertain values for the background class (class 0).
    The goal is to use information from the image to assign certain background masks for
    some pixels for which the square of a certain radius r centered at the pixel only contains
    pixels of color [0] (black) in the image.
    """
    mask_new = mask.clone()
    C, H, W = mask.shape
    for i in range(H):
        for j in range(W):
            if not neighborhood_has_colors(i,j,image, r=r, threshold = threshold):
                mask_new[0,i,j] = 1
                mask_new[1:,i,j] = 0
    return mask_new

def brute_force_filtering(image):
    filtered_image = image.clone()  # Clone to avoid modifying original
    filtered_image[filtered_image < 0.1] = torch.nan  # Use NaN to remove from display
    plt.imshow(filtered_image.squeeze(0).numpy(), cmap='gray')
    plt.axis('off')
    plt.show()

# Plot the evolution of the val and train losses
def plot_train_val_loss(train_losses,val_losses):
    plt.figure()
    plt.plot(train_losses, label="train loss")
    plt.plot(val_losses, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()

# Visualize predictions on test
def visualize_some_preds(loader, dataset, model, batch_size):
    images = next(iter(loader))
    images = images.to(DEVICE)
    logits = model(images)
    preds = torch.argmax(logits, dim=1)
    for i in range(batch_size):
        img = dataset[i]
        pred = preds[i]
        visualize_image(img)
        visualize_pred(pred)