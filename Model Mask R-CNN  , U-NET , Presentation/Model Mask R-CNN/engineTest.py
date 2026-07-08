# engineTest.py

import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import os

from dataset import MelanomaDataset  # Your custom dataset class
from engine import evaluate, calculate_iou  # Reuse the evaluation functions

# Set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Paths
test_images_dir = "./ISBI2016_ISIC_Part1_Test_Data"
test_masks_dir = "./ISBI2016_ISIC_Part1_Test_GroundTruth"
output_masks_dir = "./output_masks"

# Create output directory if not exists
os.makedirs(output_masks_dir, exist_ok=True)

# Transformations
transform = T.Compose([T.ToTensor()])

# Test dataset and DataLoader
test_dataset = MelanomaDataset(test_images_dir, test_masks_dir, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Load model
weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
model = maskrcnn_resnet50_fpn(weights=weights)

# Adjust model for 2 classes (background + melanoma)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)

in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, 2)

# Load your trained weights
model.load_state_dict(torch.load("model_maskrcnn.pth"))
model.to(device)
model.eval()

# Function to save predicted masks
def save_masks(masks, idx):
    for i, mask in enumerate(masks):
        mask = mask.squeeze(0).cpu().numpy()
        mask = np.where(mask > 0.5, 1, 0)
        
        plt.imshow(mask, cmap='gray')
        plt.axis('off')
        mask_path = os.path.join(output_masks_dir, f'mask_{idx}_{i}.png')
        plt.savefig(mask_path, bbox_inches='tight', pad_inches=0)
        plt.close()

# Function to calculate metrics
def calculate_metrics(pred_masks, true_masks):
    pred_masks = (pred_masks > 0.5).float()
    true_masks = (true_masks > 0.5).float()
    
    pred_flat = pred_masks.view(-1).cpu().numpy()
    true_flat = true_masks.view(-1).cpu().numpy()
    
    accuracy = accuracy_score(true_flat, pred_flat)
    cm = confusion_matrix(true_flat, pred_flat)
    
    return accuracy, cm

# Main evaluation loop
def evaluate_and_generate_masks(model, test_loader, device):
    all_accuracies = []
    all_confusion_matrices = []
    
    with torch.no_grad():
        for idx, (images, targets) in enumerate(test_loader):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            predictions = model(images)
            
            pred_masks = predictions[0]['masks']
            true_masks = targets[0]['masks']
            
            if pred_masks.shape[0] == 0:
                print(f"No masks predicted for sample {idx}")
                continue

            # Only take the first mask for simplicity
            accuracy, cm = calculate_metrics(pred_masks[0], true_masks[0])
            all_accuracies.append(accuracy)
            all_confusion_matrices.append(cm)
            
            # Save masks
            save_masks(pred_masks, idx)
    
    if all_accuracies:
        mean_accuracy = np.mean(all_accuracies)
        mean_cm = np.sum(all_confusion_matrices, axis=0)
        
        print(f"\nEvaluation Results:")
        print(f"Mean Accuracy: {mean_accuracy:.4f}")
        print(f"Mean Confusion Matrix:\n{mean_cm}")
    else:
        print("No predictions were made!")

# Run everything
evaluate_and_generate_masks(model, test_loader, device)
