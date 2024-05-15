import numpy as np

# Activation functions
def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of the sigmoid function"""
    return sigmoid(x) * (1 - sigmoid(x))

print(sigmoid(2))
print(sigmoid_derivative(2))
# Initialize parameters
# Number of neurons in each layer
input_layer_neurons = 2   # Input layer (number of features)
hidden_layer_neurons = 3  # Hidden layer
output_neurons = 1        # Output layer

# Initialize weights and biases with random values
# Weights for connections between input layer and hidden layer
wh = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
# Biases for neurons in hidden layer
bh = np.random.uniform(size=(1, hidden_layer_neurons))
# Weights for connections between hidden layer and output layer
wout = np.random.uniform(size=(hidden_layer_neurons, output_neurons))
# Biases for neurons in output layer
bout = np.random.uniform(size=(1, output_neurons))

# Forward propagation
def forward_propagation(X):
    """
    Perform forward propagation through the network.
    
    X: Input data
    Returns: activations of hidden layer and final output
    """
    # Input to hidden layer
    hidden_layer_input = np.dot(X, wh) + bh
    # Activation of hidden layer
    hidden_layer_activation = sigmoid(hidden_layer_input)
    # Hidden layer to output layer
    output_layer_input = np.dot(hidden_layer_activation, wout) + bout
    # Final output
    output = sigmoid(output_layer_input)
    return hidden_layer_activation, output

# Backpropagation
def backpropagation(X, y, hidden_layer_activation, output):
    """
    Perform backpropagation to update weights and biases.
    
    X: Input data
    y: True labels
    hidden_layer_activation: Activation of hidden layer from forward propagation
    output: Predicted output from forward propagation
    """
    global wh, bh, wout, bout

    # Calculate error in output
    output_error = y - output
    # Calculate delta (gradient) for output layer
    output_delta = output_error * sigmoid_derivative(output)

    # Calculate error in hidden layer
    hidden_layer_error = output_delta.dot(wout.T)
    # Calculate delta (gradient) for hidden layer
    hidden_layer_delta = hidden_layer_error * sigmoid_derivative(hidden_layer_activation)

    # Update weights and biases for output layer
    wout += hidden_layer_activation.T.dot(output_delta)
    bout += np.sum(output_delta, axis=0, keepdims=True)
    # Update weights and biases for hidden layer
    wh += X.T.dot(hidden_layer_delta)
    bh += np.sum(hidden_layer_delta, axis=0, keepdims=True)

# Training the neural network
def train(X, y, epochs):
    """
    Train the neural network using the given data.
    
    X: Input data
    y: True labels
    epochs: Number of training iterations
    """
    for epoch in range(epochs):
        # Perform forward propagation
        hidden_layer_activation, output = forward_propagation(X)
        # Perform backpropagation and update weights/biases
        backpropagation(X, y, hidden_layer_activation, output)
        # Print loss every 1000 epochs
        if epoch % 1000 == 0:
            loss = np.mean(np.square(y - output))
            print(f'Epoch {epoch} Loss: {loss}')

# Example dataset (XOR problem)
# Input data (X) and corresponding labels (y)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Train the neural network
#train(X, y, 10000)

# Predict function
def predict(X):
    """
    Predict the output for given input data.
    
    X: Input data
    Returns: Predicted output
    """
    _, output = forward_propagation(X)
    return output

# Test the network
#print("Predictions:")
#rint(predict(X))
