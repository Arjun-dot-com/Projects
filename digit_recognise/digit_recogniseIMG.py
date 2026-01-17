import numpy as np
np.random.seed(0)

import os
from PIL import Image

def read_image(folder_path):
    images = []
    labels = []
    dir_path = folder_path

    # Sorting to ensure folders 0-9 are processed in order
    folders = sorted([f for f in os.listdir(dir_path) if not f.startswith('.')])

    print(f"Starting image loading from: {dir_path}")
    for folder in folders:
        print(f"  Reading folder: {folder}")
        folder_full_path = os.path.join(dir_path, folder)
        for file in os.listdir(folder_full_path):
            if file.startswith('.'): continue
            image_path = os.path.join(folder_full_path, file)
            # convert('L') ensures grayscale 28x28
            image_array = np.array(Image.open(image_path).convert('L'))
            images.append(image_array)
            labels.append(int(folder))
    
    all_img = np.array(images)
    all_labels = np.array(labels)
    print(f"Finished loading {len(all_img)} images.")

    return all_img, all_labels

# 1e-7 is done so that even in the rare case that std = 0, no zero division error occurs
def normalize(arr, mean, std):
    return (arr - mean) / (std + 1e-7)

# Einsum takes the input string we give and changes the matrix according to the instruction in the string
def flatten(arr):
    output = arr.reshape((arr.shape[0], arr.shape[1] * arr.shape[2]))
    output = np.einsum('ij -> ji', output)
    return output

def one_hot(y, c=10):
    return np.eye(c)[y].T

class AntonMNIST():
    
    # 784 is taken for a 28 x 28 matrix
    # w1 and w2 are the weights for the 1st and 2nd layers respectively
    # b1, b2 are the biases

    def __init__(self, input_units=784, hidden_units=20):
        np.random.seed(0)
        self.w1 = np.random.randn(hidden_units, input_units) * np.sqrt(2/input_units)
        self.b1 = np.zeros((hidden_units, 1))
        # Updated w2 to (10, hidden_units) for multi-class
        self.w2 = np.random.randn(10, hidden_units) * np.sqrt(2/hidden_units)
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
        z1 = np.dot(self.w1, input) + self.b1
        a1 = self.relu(z1)
        z2 = np.dot(self.w2, a1) + self.b2
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
    
    def calculate_loss(self, logits, labels_oh):
        m = labels_oh.shape[1]
        # Categorical Cross-Entropy Loss
        loss = -np.sum(labels_oh * np.log(logits + 1e-8)) / m
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
        self.gradients["dz2"] = dz2
        self.gradients["dz1"] = dz1

        self.gradients["dw1"] = dw1
        self.gradients["db1"] = db1
        self.gradients["dw2"] = dw2
        self.gradients["db2"] = db2

    def update_parameters(self, lr):
        self.w1 -= lr * self.gradients["dw1"]
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
        batches.append((x[:, start:end], y_oh[:, start:end]))
        i += batch_size

    return batches

def accuracy(logits, labels):
    predictions = np.argmax(logits, axis=0)
    return np.mean(predictions == labels)

if __name__ == "__main__":
    print("Loading Training Images...")
    x_train_raw, y_train_raw = read_image('./train/')
    print("Loading Test Images...") 
    x_test_raw, y_test_raw = read_image('./test/')
    
    print("Calculating dataset mean and standard deviation...")
    dataset_mean, dataset_std = x_train_raw.mean(), x_train_raw.std()
    
    print("Normalizing and Flattening data...")
    x_train_scaled = normalize(x_train_raw, dataset_mean, dataset_std)
    x_test_scaled = normalize(x_test_raw, dataset_mean, dataset_std)

    x_train_scaled = flatten(x_train_scaled)
    x_test_scaled = flatten(x_test_scaled)

    print("Converting labels to One-Hot encoding...")
    y_train_oh = one_hot(y_train_raw)

    print("Shuffling the training data...")
    indices = np.arange(x_train_scaled.shape[1])
    np.random.shuffle(indices)

    x_train_scaled_shuffled = x_train_scaled[:, indices]
    y_train_oh_shuffled = y_train_oh[:, indices]
    y_train_raw_shuffled = y_train_raw[indices]

    # Hyperparameters
    learning_rate = 0.1
    batch_size = 32

    print("Generating mini-batches...")
    batches = get_batches(batch_size, x_train_scaled_shuffled, y_train_oh_shuffled)
    model = AntonMNIST(hidden_units=128)

    print(f"Starting Training: {len(batches)} batches per epoch.")
    num_epochs = 5
    for epoch in range(num_epochs):
        for batch_idx, (features, targets) in enumerate(batches):
            if np.any(np.isnan(features)):
                print(f"NaN detected in features at batch {batch_idx}")
                break

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
                
    print("Training Complete. Evaluating on Test Set...")
    test_output_probs = model.forward(x_test_scaled)
    loss_test = model.calculate_loss(test_output_probs, one_hot(y_test_raw))
    acc_test = accuracy(test_output_probs, y_test_raw)
    print(f"Loss on testing dataset: {loss_test:.4f}")
    print(f"Accuracy on testing dataset: {acc_test * 100:.2f}%")