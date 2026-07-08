import torch
from PIL import Image
import matplotlib.pyplot as plt
from model import get_model_instance_segmentation
from torchvision import transforms

# Load the trained model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model_instance_segmentation(num_classes=2)
model.load_state_dict(torch.load("mask_rcnn_epoch_10.pth"))
model.to(device)
model.eval()

# Load image
image = Image.open("data/images/img1.jpg").convert("RGB")
transform = transforms.ToTensor()
image_tensor = transform(image).unsqueeze(0).to(device)

# Run the model
with torch.no_grad():
    prediction = model(image_tensor)

# Display the masks
masks = prediction[0]['masks'].cpu().numpy()
plt.imshow(masks[0])
plt.show()
