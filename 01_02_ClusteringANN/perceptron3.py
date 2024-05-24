# Redefine necessary components and train the model

import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv('irisBinary.csv')

# Map class labels to binary values
data['Iris Class'] = data['Iris Class'].map({'Iris-setosa': 0, 'Iris-versicolor': 1})

# Split dataset into features and labels
X = data[['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']].values
y = data['Iris Class'].values.reshape(-1, 1)

# Standardize the features
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Sigmoid derivative function
def sigmoid_derivative(x):
    return x * (1 - x)

# Set hyperparameters
input_neurons = 4  # Number of attributes
output_neurons = 1  # Representing the class
epochs = 200

# Store initial weights and biases for reproducibility
initial_weight = np.random.uniform(size=(input_neurons, output_neurons))
initial_bias = np.random.uniform(size=(1, output_neurons))

# Forward propagation
def forward_propagation(X, weight, bias):
    output_layer_input = np.dot(X, weight) + bias
    output = sigmoid(output_layer_input)
    return output

# Backpropagation and training
def train(X, y, epochs, learning_rate, initial_weight, initial_bias):
    weight = np.copy(initial_weight)
    bias = np.copy(initial_bias)
    loss_history = []
    
    for epoch in range(epochs):
        output = forward_propagation(X, weight, bias)
        error = y - output
        delta = error * sigmoid_derivative(output)
        
        weight += np.dot(X.T, delta) * learning_rate
        bias += np.sum(delta, axis=0, keepdims=True) * learning_rate
        
        loss = np.mean(np.square(y - output))
        loss_history.append(loss)
    
    return weight, bias, loss_history

# Predict function
def predict(X, weight, bias):
    output = forward_propagation(X, weight, bias)
    return output

# Training with different learning rates
learning_rates = [0.001, 0.01, 0.1, 0.5, 1.0]
losses = {}

for lr in learning_rates:
    weight, bias, loss_history = train(X, y, epochs, lr, initial_weight, initial_bias)
    losses[lr] = loss_history
    print(loss_history)

