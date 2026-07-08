# import torch
# import torchvision
# from torchvision.models.detection import maskrcnn_resnet50_fpn
# from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
# from dataset import MelanomaDataset  # Your custom dataset class
# import torchvision.transforms as T
# import os
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.metrics import accuracy_score, confusion_matrix  # <-- ADD THIS


# # Set device
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# # Paths for the test dataset
# test_images_dir = "./ISBI2016_ISIC_Part1_Test_Data"
# test_masks_dir = "./ISBI2016_ISIC_Part1_Test_GroundTruth"
# output_masks_dir = "./output_masks"

# # Create the output directory if it doesn't exist
# os.makedirs(output_masks_dir, exist_ok=True)

# # Transformation
# transform = T.Compose([T.ToTensor()])

# # Create test dataset and DataLoader
# test_dataset = MelanomaDataset(test_images_dir, test_masks_dir, transform=transform)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# # Load model architecture with the weights used during training
# weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
# model = maskrcnn_resnet50_fpn(weights=weights)

# # Update the model to match the 2-class problem (background + melanoma)
# in_features = model.roi_heads.box_predictor.cls_score.in_features
# model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)  # 2 classes (background + melanoma)

# # Update the mask predictor
# in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
# hidden_layer = 256
# model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, 2)

# # Load saved model weights
# model.load_state_dict(torch.load("model_maskrcnn.pth"))
# model.to(device)  # Move model to GPU or CPU
# model.eval()  # Set model to evaluation mode


# # Function to save predicted masks
# def save_masks(masks, idx):
#     for i, mask in enumerate(masks):
#         mask = mask.squeeze(0).cpu().numpy()
#         mask = np.where(mask > 0.5, 1, 0)  # Thresholding the mask
        
#         # Save the mask as an image
#         plt.imshow(mask, cmap='gray')
#         mask_path = os.path.join(output_masks_dir, f'mask_{idx}_{i}.png')
#         plt.savefig(mask_path)
#         plt.close()


# # Removed the calculate_metrics function


# # Main evaluation loop
# def evaluate_and_generate_masks(model, test_loader, device):
#     with torch.no_grad():
#         for idx, (images, targets) in enumerate(test_loader):
#             images = [image.to(device) for image in images]
#             targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
#             # Model inference
#             prediction = model(images)
            
#             # Get the predicted masks (for evaluation)
#             pred_masks = prediction[0]['masks']
#             true_masks = targets[0]['masks']
            
#             # Removed metrics calculation
#             # accuracy, cm = calculate_metrics(pred_masks, true_masks)  # This line is removed
            
#             # Save predicted masks
#             save_masks(pred_masks, idx)
    
#     # No accuracy and confusion matrix calculations now


# # Run the evaluation and mask generation
# evaluate_and_generate_masks(model, test_loader, device)




















import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from dataset import MelanomaDataset  # Your custom dataset class
import torchvision.transforms as T
import os
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# Device setup
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Paths
test_images_dir = "./ISBI2016_ISIC_Part1_Test_Data"
test_masks_dir = "./ISBI2016_ISIC_Part1_Test_GroundTruth"

# Transform
transform = T.Compose([T.ToTensor()])

# Dataset and DataLoader
test_dataset = MelanomaDataset(test_images_dir, test_masks_dir, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Model loading
weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
model = maskrcnn_resnet50_fpn(weights=weights)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)

in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
    in_features_mask, hidden_layer, 2)

model.load_state_dict(torch.load("model_maskrcnn.pth", map_location=device))
model.to(device)
model.eval()

# Evaluation
def evaluate(model, loader, device):
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)

            # Check if 'masks' are available
            if 'masks' in outputs[0] and len(outputs[0]['masks']) > 0:
                pred_mask = outputs[0]['masks'][0, 0] > 0.5
            else:
                # If no masks are predicted, skip this image
                continue

            true_mask = targets[0]['masks'][0] > 0.5

            pred_mask = pred_mask.cpu().numpy().astype(np.uint8).flatten()
            true_mask = true_mask.cpu().numpy().astype(np.uint8).flatten()

            all_preds.extend(pred_mask)
            all_trues.extend(true_mask)

    all_preds = np.array(all_preds)
    all_trues = np.array(all_trues)

    # Metrics
    cm = confusion_matrix(all_trues, all_preds).ravel()
    if len(cm) == 4:
        tn, fp, fn, tp = cm
    else:
        tn = fp = fn = 0
        tp = cm[0]

    iou = tp / (tp + fp + fn + 1e-7)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-7)
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)

    print(f"IoU: {iou:.4f}")
    print(f"Dice Coefficient: {dice:.4f}")
    print(f"Pixel Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    # Plot confusion matrix
    plot_confusion_matrix(np.array([tn, fp, fn, tp]))


def plot_confusion_matrix(cm):
    tn, fp, fn, tp = cm
    conf_matrix = np.array([[tn, fp], [fn, tp]])

    labels = ['Background (0)', 'Lesion (1)']
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Pixel-level Confusion Matrix')
    plt.tight_layout()
    plt.show()

# Run evaluation
evaluate(model, test_loader, device)
