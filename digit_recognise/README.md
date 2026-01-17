# Digit Recognizer: Neural Network from Scratch

This repository contains a custom implementation of a 2-layer Neural Network built entirely from scratch using **NumPy**. The project demonstrates the fundamental mechanics of Deep Learning by solving the MNIST digit recognition challenge without the use of high-level frameworks like TensorFlow or PyTorch.

## 🚀 Project Overview
The project features two distinct implementations of the `AntonMNIST` model:
1. **Image Version (`img_version.py`)**: Processes raw grayscale images (28x28) organized in class-specific directories.
2. **CSV Version (`csv_version.py`)**: Optimized for [Kaggle's Digit Recognizer Competition](https://www.kaggle.com/competitions/digit-recognizer), handling flattened pixel data and generating submission files.



## 🧠 Technical Architecture
The model is a Fully Connected (Dense) Neural Network with the following specifications:

* **Input Layer**: 784 units (corresponding to 28x28 pixel images).
* **Hidden Layer**: 128 units with **ReLU** (Rectified Linear Unit) activation.
* **Output Layer**: 10 units with **Softmax** activation for multi-class probability distribution.
* **Initialization**: He Initialization ($w = \text{randn} \cdot \sqrt{2/n}$) to maintain stable variance across layers.

## 🛠️ Key Features
* **Manual Backpropagation**: Every gradient calculation (weights and biases) is derived and implemented using NumPy matrix operations.
* **Optimization**: Uses Mini-batch Gradient Descent to balance training speed and convergence stability.
* **Advanced NumPy**: Leverages `np.einsum` for efficient matrix reshaping and transposition.
* **Preprocessing**: Includes Z-score normalization (mean/std) and One-Hot encoding for categorical labels.



## 📊 Performance
* **Kaggle Score**: **0.96546**
* **Methodology**: Achieved through 10 epochs of training with a learning rate of 0.1 and batch size of 64.

## 📂 How to Run

### Prerequisites
```bash
pip install numpy pandas pillow