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
import json
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from utils import PartialCrossEntropyLoss, SimulatedRemoteSensingDataset, load_dataset

class ExperimentConfig:
    def __init__(self, label_sparsity, backbone_type, learning_rate=0.001, batch_size=8):
        self.label_sparsity = label_sparsity
        self.backbone_type = backbone_type
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def get_experiment_name(self):
        return f"sparsity_{self.label_sparsity}_backbone_{self.backbone_type}_{self.timestamp}"

class SegmentationModel(nn.Module):
    def __init__(self, num_classes=7, backbone_type='resnet18'):
        super(SegmentationModel, self).__init__()
        if backbone_type == 'resnet18':
            try:
                self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            except Exception as e:
                print("Warning: Could not load pretrained weights. Using random initialization.")
                self.backbone = models.resnet18(weights=None)
        else:  # resnet50
            try:
                self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            except Exception as e:
                print("Warning: Could not load pretrained weights. Using random initialization.")
                self.backbone = models.resnet50(weights=None)
            
        # Remove the final fully connected layer and pooling
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Add segmentation head with upsampling
        in_channels = 512 if backbone_type == 'resnet18' else 2048
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )
        
        # Add upsampling layers to restore original dimensions
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
    
    def forward(self, x):
        # Store input size for upsampling
        input_size = x.size()[2:]
        
        # Forward through backbone and segmentation head
        features = self.backbone(x)  # [B, 512/2048, H/32, W/32]
        x = self.segmentation_head(features)  # [B, num_classes, H/32, W/32]
        
        # Upsample to original input size
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    model = model.to(device)
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_idx, (images, targets, masks) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training')):
            images, targets, masks = images.to(device), targets.to(device), masks.to(device)
            
            # Debug prints for first batch
            if batch_idx == 0 and epoch == 0:
                print(f"\nInput shapes:")
                print(f"Images: {images.shape}")
                print(f"Targets: {targets.shape}")
                print(f"Masks: {masks.shape}")
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # Debug prints for first batch
            if batch_idx == 0 and epoch == 0:
                print(f"Output shape: {outputs.shape}")
            
            loss = criterion(outputs, targets, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets, masks in val_loader:
                images, targets, masks = images.to(device), targets.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets, masks)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'checkpoints/{config.get_experiment_name()}_best.pth')
    
    return history

def run_experiment(config):
    # Create experiment directory
    os.makedirs('experiments', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # Load data
    train_images, train_masks = load_dataset(split='train')
    
    # Create validation set from training data (20% split)
    train_size = int(0.8 * len(train_images))
    val_images = train_images[train_size:]
    val_masks = train_masks[train_size:]
    train_images = train_images[:train_size]
    train_masks = train_masks[:train_size]
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets with specified label sparsity
    train_dataset = SimulatedRemoteSensingDataset(train_images, train_masks, transform)
    val_dataset = SimulatedRemoteSensingDataset(val_images, val_masks, transform)
    
    # Modify label sparsity in the dataset
    train_dataset.label_sparsity = config.label_sparsity
    val_dataset.label_sparsity = config.label_sparsity
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Initialize model and training components
    model = SegmentationModel(num_classes=7, backbone_type=config.backbone_type)
    criterion = PartialCrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=device)
    
    # Save experiment results
    results = {
        'config': vars(config),
        'history': history,
        'best_val_loss': min(history['val_loss'])
    }
    
    with open(f'experiments/{config.get_experiment_name()}_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

def plot_results(experiments):
    # Plot training curves
    plt.figure(figsize=(12, 6))
    for exp_name, results in experiments.items():
        plt.plot(results['history']['train_loss'], label=f'{exp_name} (Train)')
        plt.plot(results['history']['val_loss'], '--', label=f'{exp_name} (Val)')
    
    plt.title('Training and Validation Loss Across Experiments')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('experiments/training_curves.png')
    plt.close()
    
    # Plot final performance comparison
    plt.figure(figsize=(10, 6))
    final_performance = {exp_name: results['best_val_loss'] for exp_name, results in experiments.items()}
    plt.bar(final_performance.keys(), final_performance.values())
    plt.title('Final Validation Loss by Experiment')
    plt.xticks(rotation=45)
    plt.ylabel('Best Validation Loss')
    plt.tight_layout()
    plt.savefig('experiments/final_performance.png')
    plt.close()

if __name__ == "__main__":
    # Define experiment configurations
    configs = [
        ExperimentConfig(label_sparsity=0.2, backbone_type='resnet18'),
        ExperimentConfig(label_sparsity=0.2, backbone_type='resnet50'),
        ExperimentConfig(label_sparsity=0.5, backbone_type='resnet18'),
        ExperimentConfig(label_sparsity=0.5, backbone_type='resnet50'),
    ]
    
    # Run experiments
    experiments = {}
    for config in configs:
        print(f"\nRunning experiment: {config.get_experiment_name()}")
        results = run_experiment(config)
        experiments[config.get_experiment_name()] = results
    
    # Plot results
    plot_results(experiments) 
