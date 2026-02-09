# CIFAR-10 CNN Classifier

This is a PyTorch implementation of a CNN for classifying images from the CIFAR-10 dataset. 

## Model
The network has 3 conv layers followed by max pooling and 2 linear layers. It takes 32x32 RGB images as input.

## How to run
To get started, install the requirements:
`pip install -r requirements.txt`

Then you can just run the `CIFAR10_CNN_Classifier.ipynb` notebook. The dataset downloads itself automatically into a `/data` folder.

## Details
- Trained for 20 epochs.
- Uses SGD with 0.9 momentum.
- Mixed precision (AMP) is enabled if a GPU is detected.
- Final predictions are visualized at the end of the notebook.