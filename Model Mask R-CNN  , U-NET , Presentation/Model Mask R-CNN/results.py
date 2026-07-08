import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
import torchvision.transforms as T
from dataset import MelanomaDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

# --- Setup ---
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load test dataset
test_images_dir = "./ISBI2016_ISIC_Part1_Test_Data"
test_masks_dir = "./ISBI2016_ISIC_Part1_Test_GroundTruth"

transform = T.Compose([
    T.ToTensor(),
])

test_dataset = MelanomaDataset(test_images_dir, test_masks_dir, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Load model
weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
model = maskrcnn_resnet50_fpn(weights=weights)

# Adjust for 2 classes (background, melanoma)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)

in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, 2)

model.load_state_dict(torch.load("model_maskrcnn.pth", map_location=device))
model.to(device)
model.eval()

# --- Evaluate ---
y_true = []
y_pred = []

ious = []
precisions = []
recalls = []
accuracies = []

@torch.no_grad()
def evaluate_model(model, data_loader, device):
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for output, target in zip(outputs, targets):
            if len(output["masks"]) == 0:
                continue  # No prediction

            pred_mask = output["masks"][0, 0] > 0.5
            pred_mask = pred_mask.cpu()

            true_mask = target["masks"][0] > 0

            # Resize if necessary
            if pred_mask.shape != true_mask.shape:
                pred_mask = torch.nn.functional.interpolate(pred_mask.unsqueeze(0).unsqueeze(0).float(), size=true_mask.shape, mode='bilinear', align_corners=False)
                pred_mask = pred_mask.squeeze() > 0.5

            # Calculate metrics
            intersection = (pred_mask & true_mask).float().sum()
            union = (pred_mask | true_mask).float().sum()
            iou = (intersection + 1e-6) / (union + 1e-6)
            ious.append(iou.item())

            pred_mask_np = pred_mask.flatten().numpy()
            true_mask_np = true_mask.flatten().numpy()

            precision = precision_score(true_mask_np, pred_mask_np, zero_division=0)
            recall = recall_score(true_mask_np, pred_mask_np, zero_division=0)
            accuracy = accuracy_score(true_mask_np, pred_mask_np)

            precisions.append(precision)
            recalls.append(recall)
            accuracies.append(accuracy)

            # For confusion matrix
            y_true.append(true_mask_np.sum() > 0)  # 1 if lesion exists, else 0
            y_pred.append(pred_mask_np.sum() > 0)  # 1 if model predicts lesion, else 0

# Run Evaluation
evaluate_model(model, test_loader, device)

# --- Print Results ---
print(f"Evaluated on {len(y_true)} images")
print(f"Mean IoU: {np.mean(ious):.4f}")
print(f"Mean Precision: {np.mean(precisions):.4f}")
print(f"Mean Recall: {np.mean(recalls):.4f}")
print(f"Mean Accuracy: {np.mean(accuracies):.4f}")

# --- Confusion Matrix ---
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Lesion', 'Lesion'], yticklabels=['No Lesion', 'Lesion'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# --- Optionally Save Results to CSV ---
import pandas as pd

results_df = pd.DataFrame({
    "IoU": ious,
    "Precision": precisions,
    "Recall": recalls,
    "Accuracy": accuracies,
    "True_Label": y_true,
    "Predicted_Label": y_pred
})

results_df.to_csv("evaluation_results.csv", index=False)
print("Results saved to evaluation_results.csv ✅")
