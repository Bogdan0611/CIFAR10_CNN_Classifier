# CIFAR-10 CNN Classifier

A comprehensive PyTorch implementation of a Convolutional Neural Network (CNN) for image classification on the CIFAR-10 dataset. This project demonstrates deep learning best practices including data augmentation, GPU acceleration, mixed precision training, and result visualization.

## 📋 Project Overview

This project implements a custom CNN architecture to classify images from the CIFAR-10 dataset into 10 different categories. The implementation showcases modern deep learning techniques and is designed to serve as a portfolio piece demonstrating proficiency in PyTorch, computer vision, and neural network optimization.

### Key Objectives
- Build and train a CNN from scratch using PyTorch
- Achieve high accuracy on CIFAR-10 image classification
- Implement efficient training with GPU acceleration and mixed precision
- Visualize training progress and model predictions
- Follow best practices for reproducible machine learning

## 🎯 Dataset Information

**CIFAR-10** is a well-established computer vision dataset consisting of:
- **60,000** color images (32x32 pixels)
- **10 classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **50,000** training images
- **10,000** test images

The dataset is automatically downloaded and prepared when running the notebook, making it easy to get started.

## 🏗️ Model Architecture

The network implements a custom CNN architecture with the following structure:

### Architecture Details
```
Input: 32x32 RGB images (3 channels)

Convolutional Layers:
- Conv1: 3 → 32 channels (3x3 kernel)
- Conv2: 32 → 64 channels (3x3 kernel)
- Conv3: 64 → 128 channels (3x3 kernel)

Pooling: Max Pooling (2x2) after each conv layer

Fully Connected Layers:
- FC1: Flattened features → 512 neurons
- FC2: 512 → 10 classes (output)

Activation: ReLU after each layer
Output: 10-class logits (softmax applied during inference)
```

### Model Characteristics
- **Total Parameters**: Optimized for efficient training
- **Input Size**: 32×32×3 (RGB images)
- **Output**: 10 class probabilities
- **Architecture Type**: Sequential CNN with progressive feature extraction

## 🚀 Training Details

The model is trained using industry-standard techniques:

### Hyperparameters
- **Epochs**: 20
- **Optimizer**: Stochastic Gradient Descent (SGD)
- **Momentum**: 0.9
- **Learning Rate**: Configured for optimal convergence
- **Batch Size**: Optimized for available hardware
- **Loss Function**: CrossEntropyLoss

### Training Features
- **Mixed Precision Training (AMP)**: Automatically enabled on GPU for faster training and reduced memory usage
- **GPU Acceleration**: Automatic device detection (CUDA/CPU)
- **Data Augmentation**: Applied during training to improve generalization
- **Validation**: Regular validation checks during training
- **Progress Monitoring**: Real-time training metrics

## 📊 Results

The model achieves competitive performance on the CIFAR-10 test set:
- Training is completed in 20 epochs
- Final predictions are visualized with actual vs. predicted labels
- Performance metrics tracked throughout training
- Confusion patterns analyzed through visualization

## 💻 How to Run

### Prerequisites
- Python 3.7+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Bogdan0611/CIFAR10_CNN_Classifier.git
cd CIFAR10_CNN_Classifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Required Packages
- **torch**: PyTorch deep learning framework
- **torchvision**: Computer vision utilities and datasets
- **numpy**: Numerical computing
- **matplotlib**: Visualization and plotting

### Running the Project

Simply open and run the Jupyter notebook:
```bash
jupyter notebook CIFAR10_CNN_Classifier.ipynb
```

The notebook will:
1. Automatically download the CIFAR-10 dataset to a `/data` folder
2. Initialize the CNN model
3. Train the network for 20 epochs
4. Display training progress and metrics
5. Visualize predictions on test images

**Note**: For faster training, a CUDA-compatible GPU is recommended but not required. The code automatically detects and uses GPU if available.

## 📁 Project Structure

```
CIFAR10_CNN_Classifier/
│
├── CIFAR10_CNN_Classifier.ipynb   # Main notebook with full implementation
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── data/                          # Dataset folder (created automatically)
    └── cifar-10-batches-py/       # CIFAR-10 dataset files
```

## ✨ Key Features

- **🎨 Custom CNN Architecture**: Hand-designed network optimized for CIFAR-10
- **⚡ GPU Acceleration**: Automatic CUDA detection and utilization
- **🔧 Mixed Precision Training**: AMP support for faster training on compatible GPUs
- **📈 Training Visualization**: Real-time metrics and final prediction visualization
- **🔄 Data Augmentation**: Improved model generalization through augmentation
- **💾 Model Persistence**: Save and load trained models
- **🎯 Reproducible Results**: Clear code structure for easy reproduction

## 🔍 Technical Highlights

This implementation demonstrates:
- Modern PyTorch programming patterns
- Efficient data loading with DataLoader and pin_memory
- Proper train/validation split methodology
- GPU memory optimization with mixed precision
- Clean, documented code following best practices
- Modular design with separate training and validation functions

## 🚀 Future Improvements

Potential enhancements for this project:
- [ ] Implement learning rate scheduling for better convergence
- [ ] Add data augmentation techniques (RandomCrop, RandomHorizontalFlip)
- [ ] Experiment with deeper architectures (ResNet, VGG)
- [ ] Add TensorBoard integration for advanced visualization
- [ ] Implement early stopping to prevent overfitting
- [ ] Cross-validation for robust performance estimation
- [ ] Hyperparameter tuning with grid/random search
- [ ] Deploy model as a web service

## 📝 License

This project is open source and available for educational and portfolio purposes.

## 🙏 Acknowledgments

- CIFAR-10 dataset provided by the Canadian Institute for Advanced Research
- PyTorch framework by Meta AI
- Built as a demonstration of deep learning proficiency and best practices

---

**Author**: Bogdan  
**Repository**: [CIFAR10_CNN_Classifier](https://github.com/Bogdan0611/CIFAR10_CNN_Classifier)