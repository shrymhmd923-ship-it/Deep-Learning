# MNIST Handwritten Digit Classification

## Overview
This project uses PyTorch to build a fully connected neural network that classifies handwritten digits from 0 to 9 using the MNIST dataset.

## Dataset
- Training images: 50,000
- Validation images: 10,000
- Test images: 10,000
- Image size: 28 × 28 pixels
- Number of classes: 10

## Model Architecture
- Flatten layer
- Linear layer: 784 → 128
- ReLU activation
- Linear layer: 128 → 64
- ReLU activation
- Output layer: 64 → 10

## Training Configuration
- Framework: PyTorch
- Loss function: CrossEntropyLoss
- Optimizer: Adam
- Learning rate: 0.001
- Epochs: 5
- Batch size: 64

## Results
- Training Accuracy: 98.13%
- Validation Accuracy: 97.07%
- Test Accuracy: 97.35%
- Test Loss: 0.0889

## Evaluation
The project evaluates the model using:
- Training and validation loss curves
- Training and validation accuracy curves
- Confusion matrix
- Sample predictions

## Project Files
- `mnist_first_model.py`: Main training and evaluation script
- `requirements.txt`: Required Python packages
- `loss_curve.png`: Loss visualization
- `accuracy_curve.png`: Accuracy visualization
- `confusion_matrix.png`: Classification performance visualization
- `sample_predictions.png`: Example predictions

## Installation
Install the required packages:

```bash
python -m pip install -r requirements.txt
```

## Run the Project
```bash
python mnist_first_model.py
```

## Future Improvements
- Add early stopping
- Compare different neural network architectures
- Build a Convolutional Neural Network (CNN)
- Analyze incorrectly classified images