import torch
import torchvision
import numpy as np
import os
from torch.utils.data import DataLoader, random_split
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights  # Import the weights enum
import torchvision.transforms as T
from dataset import MelanomaDataset
from engine import train_one_epoch, evaluate  # We'll create 'engine.py' too!

# Set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Dataset paths
images_dir = "./ISBI2016_ISIC_Part1_Training_Data"
masks_dir = "./ISBI2016_ISIC_Part1_Training_GroundTruth"

# Transform
transform = T.Compose([
    T.ToTensor(),
])

# Dataset and DataLoader
dataset = MelanomaDataset(images_dir, masks_dir, transform=transform)

# Split into train/val
torch.manual_seed(1)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Model
weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT  # Use the default weights instead of deprecated 'pretrained=True'
model = maskrcnn_resnet50_fpn(weights=weights)  # Updated model loading with weights

# Update the box predictor for the 2-class problem (background + melanoma)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)  # 2 classes (background + melanoma)

# Update the mask predictor for the 2-class problem (background + melanoma)
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
    in_features_mask,
    hidden_layer,
    2
)

model.to(device)

# Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

# Learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Train
num_epochs = 10

for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
    lr_scheduler.step()
    evaluate(model, val_loader, device)

# Save model
torch.save(model.state_dict(), "model_maskrcnn.pth")
