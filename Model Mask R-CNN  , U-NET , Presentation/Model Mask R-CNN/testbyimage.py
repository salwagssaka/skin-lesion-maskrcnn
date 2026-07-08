import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision import transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Setup
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load the model
model = maskrcnn_resnet50_fpn(weights=None)  # Don't load pretrained weights

# Change the number of classes in the model (2 classes in your case)
# 1 class (object) + background
in_features = model.roi_heads.box_predictor.cls_score.in_features

# Modify the box predictor to match the number of classes (2 in your case)
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes=2)

# Modify the mask predictor
# The `MaskRCNNPredictor` requires 3 arguments: in_channels, hidden_layer_channels, and num_classes
in_channels = model.roi_heads.mask_predictor.conv5_mask.in_channels  # This will get the in_channels from the model
model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_channels, 256, num_classes=2)

# Load weights from the checkpoint
model.load_state_dict(torch.load("model_maskrcnn.pth", map_location=device))
model.to(device)
model.eval()

# Load your image
image_path = "./testImages/test.jpg"
image = Image.open(image_path).convert("RGB")

# Preprocess image
transform = T.Compose([
    T.ToTensor(),
])
image_tensor = transform(image).unsqueeze(0).to(device)  # (1, C, H, W)

# Run inference
with torch.no_grad():
    prediction = model(image_tensor)

# Get predictions
pred_boxes = prediction[0]['boxes'].cpu().numpy()
pred_scores = prediction[0]['scores'].cpu().numpy()
pred_labels = prediction[0]['labels'].cpu().numpy()
pred_masks = prediction[0]['masks'].cpu().numpy()

# Threshold
score_threshold = 0.5
keep = pred_scores >= score_threshold

boxes = pred_boxes[keep]
scores = pred_scores[keep]
labels = pred_labels[keep]
masks = pred_masks[keep]

print(f"Detected {len(boxes)} objects.")

# Show results
fig, ax = plt.subplots(1, figsize=(12, 9))

# Show image
image_np = np.array(image)
ax.imshow(image_np)

# For each detection
for box, mask, label, score in zip(boxes, masks, labels, scores):
    # Draw bounding box
    x1, y1, x2, y2 = box
    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    ax.text(x1, y1-10, f"Class: {label} | Score: {score:.2f}", color='red', fontsize=12, backgroundcolor='white')

    # Draw mask
    mask = mask.squeeze(0)  # (1, H, W) -> (H, W)
    ax.imshow(mask, alpha=0.5, cmap='jet')  # Mask overlay

plt.axis('off')
plt.tight_layout()
plt.show()
