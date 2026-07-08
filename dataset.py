# dataset.py
import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

class MelanomaDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_paths = sorted(os.listdir(images_dir))
        self.mask_paths = sorted(os.listdir(masks_dir))
        self.transform = transform
        self.resize = transforms.Resize((512, 512))  # Better speed

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_paths[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_paths[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Grayscale

        # Resize
        image = self.resize(image)
        mask = self.resize(mask)

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        # Prepare mask
        mask = np.array(mask)
        mask = (mask > 0).astype(np.uint8)  # Binarize
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.unsqueeze(0)

        # Bounding boxes
        pos = torch.where(mask[0] > 0)
        if pos[0].numel() > 0:
            xmin = torch.min(pos[1])
            xmax = torch.max(pos[1])
            ymin = torch.min(pos[0])
            ymax = torch.max(pos[0])
            boxes = torch.tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)
            labels = torch.ones((1,), dtype=torch.int64)  # Class 1
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": mask
        }

        return image, target

    def __len__(self):
        return len(self.image_paths)
