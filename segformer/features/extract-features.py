import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
# If using HuggingFace's SegFormer model:
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# --- Configuration ---
train_images_dir = "train-images/"
output_feature_dir = "extracted_features/"
os.makedirs(output_feature_dir, exist_ok=True)

# Load the trained SegFormer model (replace with actual model path or name)
model_path = "path/to/segformer_model"  # This could be a checkpoint or a HuggingFace model ID
model = SegformerForSemanticSegmentation.from_pretrained(model_path)
model.eval()  # set model to evaluation mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define data transformations (must match those used in training)
# Example: resize to 512x512, random horizontal flip, convert to tensor, and normalize
train_transforms = transforms.Compose([
    transforms.Resize((512, 512)),              # resize to training resolution if needed
    transforms.RandomHorizontalFlip(p=0.5),     # example augmentation used in training
    transforms.ToTensor(),                      # convert to PyTorch tensor [0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # normalize as in training (ImageNet mean/std if used)
                         std=[0.229, 0.224, 0.225])
])

# Loop over images and extract features
with torch.no_grad():  # no gradient computation
    for filename in os.listdir(train_images_dir):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            continue  # skip non-image files
        img_path = os.path.join(train_images_dir, filename)
        image = Image.open(img_path).convert("RGB")
        
        # Apply the training transformations to the image
        img_tensor = train_transforms(image)            # transformed image tensor (C x H x W)
        img_tensor = img_tensor.unsqueeze(0).to(device) # add batch dimension and move to device
        
        # Forward pass through the model to get segmentation logits (features before argmax)
        outputs = model(img_tensor)  
        # If using HuggingFace model, outputs.logits is [batch, num_classes, h, w]
        logits = outputs.logits  # shape: (1, num_classes, H_feat, W_feat)
        logits = logits.cpu().numpy()[0]  # move to CPU and convert to numpy, shape: (num_classes, H_feat, W_feat)
        
        # Save feature map to .npy file with the same base name as the image
        base_name, _ = os.path.splitext(filename)
        out_path = os.path.join(output_feature_dir, f"{base_name}.npy")
        np.save(out_path, logits)
        
        # Clean up to free memory for next iteration
        del img_tensor, outputs, logits
        torch.cuda.empty_cache()
