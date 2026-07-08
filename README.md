# 🔬 Skin Lesion Segmentation with Mask R-CNN

A deep learning project for **automatic skin lesion segmentation** using Mask R-CNN with a ResNet-50 FPN backbone, trained on the ISIC 2016 challenge dataset. The model detects and segments melanoma and other skin lesions from dermoscopic images.

---

## 📌 Problem Statement

Early detection of skin cancer, especially melanoma, is critical for patient survival. Manual analysis of dermoscopic images is time-consuming and error-prone. This project trains a **Mask R-CNN model to automatically segment skin lesions**, generating pixel-level masks that can assist dermatologists in diagnosis.

---

## 📊 Dataset

- **Source:** [ISIC 2016 Challenge — Part 1 (Lesion Segmentation)](https://challenge.isic-archive.com/)
- **Train images:** [Download](https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Training_Data.zip)
- **Train masks:** [Download](https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Training_GroundTruth.zip)
- **Test images:** [Download](https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Test_Data.zip)
- **Test masks:** [Download](https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Test_GroundTruth.zip)

---

## 🧠 Model Architecture

- **Mask R-CNN** with ResNet-50 FPN backbone (pretrained on COCO)
- Fine-tuned on the ISIC 2016 skin lesion dataset
- Outputs: bounding boxes + pixel-level segmentation masks

---

## ⚙️ Training Details

| Parameter | Value |
|-----------|-------|
| Epochs | 10 |
| Evaluation set | ~180 images |
| Backbone | ResNet-50 FPN |
| Pretrained weights | COCO (`maskrcnn_resnet50_fpn_coco`) |
| Optimizer | SGD (default torchvision) |

**Loss curve (Average loss per epoch):**

| Epoch | Avg Loss |
|-------|----------|
| 1 | 0.3674 |
| 2 | 0.2324 |
| 3 | 0.2050 |
| 5 | 0.1721 |
| 8 | 0.1661 |
| 10 | 0.1655 |

---

## 📈 Results

### Validation (per epoch, ~180 images)

| Epoch | Mean IoU | Precision | Recall | Accuracy |
|-------|----------|-----------|--------|----------|
| 1 | 0.8503 | 0.9116 | 0.9348 | 0.9556 |
| 4 | 0.8700 | 0.9378 | 0.9294 | 0.9615 |
| **10** | **0.8700** | **0.9382** | **0.9287** | **0.9610** |

### Test Set (377 images)

| Metric | Score |
|--------|-------|
| **Mean IoU** | **0.8594** |
| **Mean Precision** | **0.9255** |
| **Mean Recall** | **0.9291** |
| **Mean Accuracy** | **0.9559** |

---

## 🚀 How to Run

### 1. Set up the environment

```bash
python -m venv maskrcnn_env
source maskrcnn_env/bin/activate        # Linux/macOS
# .\maskrcnn_env\Scripts\activate       # Windows

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 2. Download the dataset

Download the 4 zip files from the links above, extract them locally, then update the dataset paths inside `train.py`, `test.py`, and `engine.py`.

### 3. Train the model

```bash
python train.py
```

### 4. Evaluate on the test set

```bash
python test.py
```

### 5. Run inference on a new image

Add your image path inside `testbyimage.py`, then:

```bash
python testbyimage.py
```

---

## 📁 Project Structure

```
├── train.py              # Training + validation loop
├── test.py               # Test phase evaluation
├── testbyimage.py        # Inference on a single image
├── model.py              # Model definition
├── dataset.py            # Dataset loading & preprocessing
├── engine.py             # Training engine
├── inference.py          # Inference utilities
├── results.py            # Results computation
├── requirements.txt      # Dependencies
├── model_maskrcnn.pth    # Saved model weights
└── testImages/           # Sample test images
    ├── Melanoma.jpg
    └── test.jpg
```

---

## 🛠️ Libraries Used

- `torch`, `torchvision` — deep learning framework and Mask R-CNN
- `opencv-python` — image processing
- `pycocotools` — COCO-format evaluation metrics
- `numpy`, `pandas` — data handling

---

## 👤 Author

[GitHub](https://github.com/salwagssaka) • [LinkedIn](https://linkedin.com/in/salwagssaka3b099428a)
