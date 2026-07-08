# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Train
python train_maskrcnn.py

# Step 3: Inference
python infer.py





# Create a virtual environment
python -m venv maskrcnn_env
source maskrcnn_env/bin/activate  # Linux/macOS
# .\maskrcnn_env\Scripts\activate  # Windows

# Install PyTorch (choose your version accordingly)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Detectron2
pip install opencv-python
pip install pycocotools
pip install git+https://github.com/facebookresearch/detectron2.git

# Optional: Jupyter notebook for experiments
pip install notebook
