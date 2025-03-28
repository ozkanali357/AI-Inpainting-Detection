# AI Inpainting Detection (AaltoES 2025 Hackathon) 🏆

**Best Start Award Winner** | Binary Segmentation for Detecting AI-Manipulated Image Regions

This repository contains the solution for the AaltoES 2025 Computer Vision Hackathon, focused on detecting AI-inpainted regions in images. The project tackles a critical challenge in digital forensics: identifying manipulated content to combat misinformation. The model achieves pixel-level binary segmentation (real vs. fake) using a U-Net architecture with a ResNet50 encoder, trained on 28k+ images.

## Key Features
- **Model Architecture**: U-Net with ResNet50 backbone pretrained on ImageNet.
- **Loss Function**: Hybrid BCE + Dice Loss for robust segmentation.
- **Data Augmentation**: Albumentations pipeline (flips, rotation, normalization).
- **Optimization**: AdamW with CosineAnnealing learning rate scheduling.
- **Efficiency**: Trained for 300 epochs with early stopping and validation.

## Technical Stack
- PyTorch • Albumentations • Segmentation Models PyTorch (SMP)
- Run-Length Encoding (RLE) for submission formatting
- Multi-GPU ready and optimized inference pipeline

## Usage
1. Clone repo: `git clone https://github.com/ozkanali357/aaltoes-inpainting-detection`
2. Install dependencies: `pip install -r requirements.txt`
3. Train: `python train.py`
4. Generate predictions: `python inference.py`

Dataset: 28,101 training images with masks (256x256 RGB). Achieved **top-tier validation performance** during the hackathon, earning the "Best Start" nomination.

`requirements.txt` file:  
```
torch>=2.0.0
segmentation-models-pytorch
albumentations
opencv-python
pandas
tqdm
```  
