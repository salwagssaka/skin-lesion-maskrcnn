import torch
import torchvision
from torchvision import models

def get_model_instance_segmentation(num_classes):
    # Load a pre-trained Mask R-CNN model
    model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    
    # Get the number of input channels in the pre-trained model (should be 3 for RGB)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the classifier with a new one for our custom number of classes
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    
    # Replace the mask predictor
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features, 256, num_classes)
    
    return model
