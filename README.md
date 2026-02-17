# CIFAR-10 CNN Classifier

This is a PyTorch implementation of a CNN that classifies images from the CIFAR-10 dataset. I built it to learn more about convolutional neural networks and how to train them efficiently with GPU acceleration and mixed precision.

## What is this?

I wanted to build a neural network from scratch to classify images into 10 different categories (airplanes, cars, birds, cats, etc.). The CIFAR-10 dataset is a good starting point for this - it has 60,000 small images (32x32 pixels) split into 50,000 for training and 10,000 for testing.

The whole thing runs in a Jupyter notebook and downloads the data automatically, so you can just clone it and start training right away.

## Model Architecture

The network has a pretty straightforward architecture:
- 3 convolutional layers (3→32, 32→64, 64→128 channels) with 3x3 kernels
- Max pooling (2x2) after each conv layer
- 2 fully connected layers at the end (512 neurons, then 10 output classes)
- ReLU activations throughout

It takes in 32x32 RGB images and outputs probabilities for each of the 10 classes.

## Training

I trained the model for 20 epochs using SGD with 0.9 momentum. The code automatically uses your GPU if you have one (with mixed precision via PyTorch's AMP to speed things up), otherwise it falls back to CPU.

The training loop includes validation checks, and at the end you get some nice visualizations of how well the model is doing on test images.

## How to Run

First, install the required packages:
```bash
pip install -r requirements.txt
```

You'll need:
- torch
- torchvision
- numpy
- matplotlib

Then just open the notebook and run it:
```bash
jupyter notebook CIFAR10_CNN_Classifier.ipynb
```

The notebook will download the CIFAR-10 dataset automatically (it goes into a `/data` folder), build the model, train it, and show you some predictions at the end.

If you have a CUDA-compatible GPU, the training will be faster. But it works fine on CPU too, just takes longer.

## Project Files

```
CIFAR10_CNN_Classifier/
├── CIFAR10_CNN_Classifier.ipynb   # Main notebook
├── requirements.txt                # Dependencies
├── README.md                       # This file
└── data/                          # Created when you run it
    └── cifar-10-batches-py/       # Dataset files
```

## Things I Might Add Later

Some ideas I had for improving this:
- Learning rate scheduling
- More data augmentation (random crops, flips, etc.)
- Try other architectures like ResNet or VGG
- TensorBoard for better training visualization
- Early stopping
- Hyperparameter tuning
- Maybe deploy it as a simple web app

## Credits

The CIFAR-10 dataset is from the Canadian Institute for Advanced Research. Built with PyTorch.