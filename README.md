# FetCAT: Cross-Attention Fusion of Transformer-CNN Architecture for Fetal Brain Plane Classification with Explainability

## Overview
This repository contains a deep learning framework for classifying fetal MRI planes using a novel hybrid architecture that combines Swin Transformer with a custom AdaptiveMedCNN. The model leverages cross-attention mechanisms for effective feature fusion and is evaluated using k-fold cross-validation.

## Model Architecture

### Hybrid Components
- **Swin Transformer Base**: Pretrained on ImageNet-22K, fine-tuned for medical imaging
- **AdaptiveMedCNN**: Custom CNN with dilated convolutions and specialized medical imaging features
- **Multi-Head Cross Attention**: Fuses features from both architectures
- **Feature Fusion Transformer**: Integrates Swin and CNN features for final classification

### Key Features
- Multi-modal feature extraction from same input images
- Cross-attention mechanism for feature interaction
- Layer-wise unfreezing strategy for optimal fine-tuning
- No data augmentation for baseline performance evaluation
- Comprehensive evaluation with k-fold cross-validation

## Dataset
- **Type**: Fetal MRI planes
- **Classes**: Multiple anatomical planes (specific classes depend on dataset structure)
- **Structure**: Organized in class-wise directories
- **Preprocessing**: Center cropping, resizing to 224×224, ImageNet normalization

## Installation

### Requirements
torch>=1.9.0
torchvision>=0.10.0
transformers>=4.20.0
scikit-learn>=1.0.0
opencv-python>=4.5.0
matplotlib>=3.5.0
seaborn>=0.11.0
pandas>=1.3.0
tqdm>=4.60.0
Pillow>=8.3.0

## Usage

### Training
python without_aug_swin_cust_cnn.py

### Configuration
- **K-Fold Cross Validation**: 2 folds (configurable)
- **Batch Size**: 8
- **Learning Rate**: 1e-4 with AdamW optimizer
- **Epochs**: 10 with early stopping (patience=3)
- **Image Size**: 224×224

## Model Details

### Swin Transformer Component
- **Backbone**: Swin-Base (patch4-window7-224)
- **Fine-tuning**: Unfreeze 2/3 of layers for medical domain adaptation
- **Features**: Global contextual understanding

### AdaptiveMedCNN Component
- **Architecture**: Medical-optimized with dilated convolutions
- **Features**: Local texture and pattern recognition
- **Freezing**: First 8 layers frozen, later layers fine-tuned

### Fusion Mechanism
- **Cross Attention**: 8-head attention between Swin and CNN features
- **Projection**: CNN features projected to Swin dimension space
- **Classification**: Multi-layer perceptron with LayerNorm and dropout

## Evaluation Metrics

The model provides comprehensive evaluation including:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix visualization
- Training/Validation loss and accuracy curves
- Cross-validation results comparison
- GradCAM visualization for interpretability

## Outputs

### Generated Files
- **Models**: Best model for each fold (models_no_aug/)
- **Plots**: 
  - Training/validation metrics per fold
  - Cross-fold comparison
  - Confusion matrices
  - Average metrics across folds
- **Results**: CSV file with detailed performance metrics

### Visualization
- Real-time training progress with tqdm
- Interactive matplotlib plots
- Seaborn-based confusion matrices
- GradCAM attention maps

## Performance

The model is designed for robust evaluation with:
- K-fold cross-validation for reliable performance estimation
- Early stopping to prevent overfitting
- Comprehensive metric tracking
- Model interpretability through attention visualization

## Customization

### Modify Architecture
Change Swin Transformer variant:
model = HybridCNNSwin(num_classes=num_classes, swin_model_name="microsoft/swin-tiny-patch4-window7-224")

Adjust fusion mechanism:
self.cross_attention = MultiHeadCrossAttention(query_dim=swin_dim, num_heads=12)

### Training Parameters
Modify in main() function:
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)


## Acknowledgments

- Swin Transformer authors for the base architecture
- Medical imaging research community
- Contributors to PyTorch and Hugging Face Transformers

## Contact

For questions and contributions, please contact:
- Your Name: [suha@bup.edu.bd]

- Multi-modal data fusion (MRI + clinical data)
- Self-supervised pretraining on unlabeled fetal MRI data
- Real-time inference optimization for clinical deployment
