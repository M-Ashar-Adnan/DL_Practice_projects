# MNIST Digit Classifier using PyTorch

This repository contains an implementation of a Deep Learning model built with PyTorch to classify handwritten digits (0-9) from the classic MNIST dataset.

## 📌 Project Overview

The project defines, trains, and evaluates a Multilayer Perceptron (MLP) neural network. It includes data loading, normalization, a custom neural network class, and an evaluation phase that calculates the model's accuracy on unseen test data, alongside generating a visual plot of the model's predictions.

### Sample Predictions

Here is a snapshot of the model's predictions on the test dataset:

![Model Predictions](DigitClassifier/predictions.png)

---

## 🏗️ Model Architecture

The model is a fully connected neural network (Feedforward Neural Network) defined in the `DigitClassifier` class:

1. **Input Layer:** Flattens the 28x28 pixel grayscale images into a 1D array of 784 features.
2. **Hidden Layer 1:** Linear layer mapping 784 inputs to 128 outputs, followed by a ReLU activation function.
3. **Hidden Layer 2:** Linear layer mapping 128 inputs to 64 outputs, followed by a ReLU activation function.
4. **Output Layer:** Linear layer mapping the features to 10 output classes (representing digits 0 through 9).

## ⚙️ Hyperparameters

- **Batch Size:** 64
- **Learning Rate:** 0.001
- **Epochs:** 5
- **Optimizer:** Adam
- **Loss Function:** Cross-Entropy Loss

---

## 🛠️ Requirements

- Python 3.7+
- `torch`
- `torchvision`
- `matplotlib`
