# engine.py
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score

def calculate_iou(pred_mask, true_mask):
    intersection = (pred_mask & true_mask).float().sum((1, 2))
    union = (pred_mask | true_mask).float().sum((1, 2))
    iou = (intersection + 1e-6) / (union + 1e-6)  # Add small number to avoid division by zero
    return iou.mean().item()

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    running_loss = 0.0

    for idx, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        if torch.isnan(losses):
            print(f"Skipping batch {idx} because loss is NaN")
            continue

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

        if idx % print_freq == 0:
            print(f"Batch {idx}, Loss: {losses.item():.4f}")

    print(f"Epoch [{epoch+1}] Average Loss: {running_loss/len(data_loader):.4f}")

@torch.no_grad()
def evaluate(model, data_loader, device, iou_threshold=0.5):
    model.eval()
    total = 0
    all_ious = []
    all_precisions = []
    all_recalls = []
    all_accuracies = []

    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        outputs = model(images)

        for output, target in zip(outputs, targets):
            if len(output["masks"]) == 0:
                continue  # No prediction

            # Take best mask
            pred_mask = output["masks"][0, 0] > 0.5  # Threshold 0.5
            pred_mask = pred_mask.cpu()

            true_mask = target["masks"][0] > 0

            # Resize if needed
            if pred_mask.shape != true_mask.shape:
                pred_mask = torch.nn.functional.interpolate(pred_mask.unsqueeze(0).unsqueeze(0).float(), size=true_mask.shape, mode='bilinear', align_corners=False)
                pred_mask = pred_mask.squeeze() > 0.5

            # IoU
            iou = calculate_iou(pred_mask.unsqueeze(0), true_mask.unsqueeze(0))
            all_ious.append(iou)

            # Precision / Recall / Accuracy
            pred_mask_np = pred_mask.flatten().numpy()
            true_mask_np = true_mask.flatten().numpy()

            precision = precision_score(true_mask_np, pred_mask_np, zero_division=0)
            recall = recall_score(true_mask_np, pred_mask_np, zero_division=0)
            accuracy = accuracy_score(true_mask_np, pred_mask_np)

            all_precisions.append(precision)
            all_recalls.append(recall)
            all_accuracies.append(accuracy)

            total += 1

    if total == 0:
        print("No predictions made.")
        return

    print(f"Evaluated on {total} images")
    print(f"Mean IoU: {np.mean(all_ious):.4f}")
    print(f"Mean Precision: {np.mean(all_precisions):.4f}")
    print(f"Mean Recall: {np.mean(all_recalls):.4f}")
    print(f"Mean Accuracy: {np.mean(all_accuracies):.4f}")

