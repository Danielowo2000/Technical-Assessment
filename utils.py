import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from PIL import Image
import os

# Custom Partial Cross-Entropy Loss
class PartialCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(PartialCrossEntropyLoss, self).__init__()
    
    def forward(self, inputs, targets, mask):
        """
        inputs: Predicted logits (batch_size, num_classes, H, W)
        targets: Ground truth labels (batch_size, H, W)
        mask: Binary mask indicating labeled pixels (batch_size, H, W)
        """
        log_probs = F.log_softmax(inputs, dim=1)
        loss = -torch.gather(log_probs, dim=1, index=targets.unsqueeze(1))
        loss = loss.squeeze(1) * mask  # Apply mask
        return loss.sum() / mask.sum()  # Normalize by labeled pixels

# Simulated Dataset for Remote Sensing Segmentation
class SimulatedRemoteSensingDataset(Dataset):
    def __init__(self, images, masks, transform=None, label_sparsity=0.2):
        self.images = images
        self.masks = masks
        self.transform = transform
        self.label_sparsity = label_sparsity
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        
        # Create label mask based on sparsity
        label_mask = (torch.rand_like(mask.float()) > (1 - self.label_sparsity)).long()
        masked_labels = mask * label_mask
        
        if self.transform:
            image = self.transform(image)
        
        return image, masked_labels, label_mask

def load_dataset(split='train', img_size=(256, 256)):
    """
    Load the actual remote sensing dataset.
    Args:
        split (str): One of 'train', 'valid', or 'test'
        img_size (tuple): Target size for the images
    Returns:
        tuple: (images, masks) where images is a list of PIL Images and masks is a list of torch tensors
    """
    # Load metadata and class dictionary
    metadata = pd.read_csv('dataset/metadata.csv')
    class_dict = pd.read_csv('dataset/class_dict.csv')
    
    # Filter metadata by split and drop rows with NaN values
    split_data = metadata[metadata['split'] == split].dropna()
    
    images = []
    masks = []
    
    # Create class to index mapping
    class_to_idx = {(row.r, row.g, row.b): idx for idx, row in enumerate(class_dict.itertuples(index=False))}
    
    for _, row in split_data.iterrows():
        # Load satellite image
        img_path = os.path.join('dataset', str(row['sat_image_path']))
        mask_path = os.path.join('dataset', str(row['mask_path']))
        
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            print(f"Warning: Missing files for image {row['image_id']}")
            continue
            
        # Load and resize image
        img = Image.open(img_path).convert('RGB')
        img = img.resize(img_size, Image.BILINEAR)
        
        # Load and resize mask
        mask = Image.open(mask_path).convert('RGB')
        mask = mask.resize(img_size, Image.NEAREST)
        mask = np.array(mask)
        
        # Convert RGB mask to class indices
        mask_indices = np.zeros(img_size, dtype=np.int64)
        for (r, g, b), idx in class_to_idx.items():
            matching_pixels = (mask[..., 0] == r) & (mask[..., 1] == g) & (mask[..., 2] == b)
            mask_indices[matching_pixels] = idx
        
        images.append(img)
        masks.append(torch.from_numpy(mask_indices))
    
    if not images:
        raise ValueError(f"No valid images found for split '{split}'")
    
    return images, masks

    def __init__(self, num_classes=2):
        super(SimpleSegmentationModel, self).__init__()
        try:
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except Exception as e:
            print("Warning: Could not load pretrained weights. Using random initialization.")
            self.backbone = models.resnet18(weights=None)
            
        # Remove the final fully connected layer and pooling
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Add a more appropriate segmentation head
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.segmentation_head(features)

