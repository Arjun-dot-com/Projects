import numpy as np
np.random.seed(0)

import os
import pandas as pd # Added for Kaggle CSV loading
from PIL import Image

def read_image(folder_path):
    images = []
    labels = []
    dir_path = folder_path

    for folder in sorted(os.listdir(dir_path)):
        if folder.startswith('.'): continue
        for file in os.listdir(dir_path + f"/{folder}/"):
            if file.startswith('.'): continue
            image_path = dir_path + f"/{folder}/" + file
            image_array = np.array(Image.open(image_path).convert('L'))
            images.append(image_array)
            labels.append(int(folder))
    
    all_img = np.array(images)
    all_labels = np.array(labels)

    return all_img, all_labels

# 1e-7 is done so that even in the rare case that std = 0, no zero division error occurs
def normalize(arr, mean, std):
    return (arr - mean) / (std + 1e-7)

# Einsum takes the input string we give and changes the matrix according to the instruction in the string
def flatten(arr):
    output = arr.reshape((arr.shape[0], 784))
    output = np.einsum('ij -> ji', output)
    return output

def one_hot(y, c=10):
    return np.eye(c)[y].T

class AntonMNIST():
    
    # 784 is taken for a 28 x 28 matrix
    # w1 and w2 are the weights for the 1st and 2nd layers respectively
    # b1, b2 are the biases

    def __init__(self, input_units=784, hidden_units=128): # Increased hidden units for 10-class
        np.random.seed(0)
        self.w1 = np.random.randn(hidden_units, input_units) * np.sqrt(2/input_units)
        self.b1 = np.zeros((hidden_units, 1))
        self.w2 = np.random.randn(10, hidden_units) * np.sqrt(2/hidden_units) # Output 10 for digits 0-9
        self.b2 = np.zeros((10, 1))

        # Storing activations for backprop
        self.activations = {}

        # Storing gradients for the update
        self.gradients = {}

    def relu(self, input):
        # Rectified Linear Unit (ReLU) activation function, which outputs the input directly if positive, or zero if negative
        input_copy = np.copy(input)
        input_copy[(input_copy < 0)] = 0
        return input_copy
        
    # The derivative of the ReLU activation function is taken in neural networks to facilitate the backpropagation process, which calculates gradients necessary for optimizing network weights and biases.
    # Backpropagation is used to efficiently calculating how much each weight and bias contributes to the overall error (loss) by propagating error signals backward from the output to the input layer
     
    def relu_derivative(self, input):
        relu_grad = np.copy(input)
        relu_grad[(relu_grad >= 0)] = 1
        relu_grad[(relu_grad < 0)] = 0
        return relu_grad
        
    def forward(self, input):
        z1 = np.dot(self.w1, input) + self.b1 # Fixed: Addition
        a1 = self.relu(z1)
        z2 = np.dot(self.w2, a1) + self.b2 # Fixed: Addition
        # Softmax for multi-class classification
        exp_z2 = np.exp(z2 - np.max(z2, axis=0, keepdims=True))
        a2 = exp_z2 / np.sum(exp_z2, axis=0, keepdims=True)

        # Activations
        self.activations["a0"] = input
        self.activations["z1"] = z1
        self.activations["a1"] = a1
        self.activations["z2"] = z2
        self.activations["a2"] = a2

        return a2
    
    def calculate_loss(self, probs, labels_oh):
        m = labels_oh.shape[1]
        # Categorical Cross-Entropy Loss
        loss = -np.sum(labels_oh * np.log(probs + 1e-8)) / m
        return loss
    
    def calculate_gradients(self, labels_oh):
        m = labels_oh.shape[1]
        dz2 = self.activations["a2"] - labels_oh
        dw2 = 1/m * np.dot(dz2, self.activations["a1"].T)
        db2 = 1/m * np.sum(dz2, axis=1, keepdims=True)
        dz1 = np.dot(self.w2.T, dz2) * self.relu_derivative(self.activations["z1"])
        dw1 = 1/m * np.dot(dz1, self.activations["a0"].T)
        db1 = 1/m * np.sum(dz1, axis=1, keepdims=True)

        # Not exactly necessary, but useful to debug
        self.gradients["dw1"] = dw1
        self.gradients["db1"] = db1
        self.gradients["dw2"] = dw2
        self.gradients["db2"] = db2

    def update_parameters(self, lr):
        self.w1 -= lr * self.gradients["dw1"] # Fixed: Subtraction
        self.b1 -= lr * self.gradients["db1"]
        self.w2 -= lr * self.gradients["dw2"]
        self.b2 -= lr * self.gradients["db2"]
        
def get_batches(batch_size, x, y_oh):
    batches = []
    m = y_oh.shape[1]

    i = 0
    while i < m:
        start = i
        end = min(i + batch_size, m)
        batches.append((x[:, start:end], y_oh[:, start:end])) # Fixed: Slice syntax
        i += batch_size

    return batches

def accuracy(probs, labels):
    predictions = np.argmax(probs, axis=0)
    return np.mean(predictions == labels)

if __name__ == "__main__":
    # Loading CSVs for Kaggle
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    y_train_raw = train_df['label'].values
    x_train_raw = train_df.drop('label', axis=1).values
    x_test_raw = test_df.values

    # Preprocessing
    x_train_scaled = x_train_raw.T / 255.0
    x_test_scaled = x_test_raw.T / 255.0
    y_train_oh = one_hot(y_train_raw)

    # Hyperparameters
    learning_rate = 0.1
    batch_size = 64
    num_epochs = 10

    batches = get_batches(batch_size, x_train_scaled, y_train_oh)
    model = AntonMNIST()

    # Plotting everything
    for epoch in range(num_epochs):
        for batch_idx, (features, targets) in enumerate(batches):
            # Forward pass
            output_probs = model.forward(features)
            loss = model.calculate_loss(output_probs, targets)

            # Backward pass
            model.calculate_gradients(targets)
            model.update_parameters(lr=learning_rate)

            # Logging
            if batch_idx % 100 == 0:
                print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
                      f" | Batch {batch_idx:03d}/{len(batches):03d}"
                      f" | Train Loss: {loss:.4f}")
                
    # Prepare Submission
    test_probs = model.forward(x_test_scaled)
    test_predictions = np.argmax(test_probs, axis=0)
    
    submission = pd.DataFrame({
        "ImageId": np.arange(1, len(test_predictions) + 1),
        "Label": test_predictions
    })
    submission.to_csv('submission.csv', index=False)
    print("Submission file saved as submission.csv")