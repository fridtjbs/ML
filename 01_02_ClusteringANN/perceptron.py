import numpy as np
import pandas as pd

# Load dataset
data = pd.read_csv('irisBinary.csv')

# Map class labels to binary values
data['Iris Class'] = data['Iris Class'].map({'Iris-setosa': 0, 'Iris-versicolor': 1})

# Split dataset into features and labels
X = data[['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']].values
y = data['Iris Class'].values.reshape(-1, 1)

#   example iris attribute input
#   [5.1 3.5 1.4 0.2]

#   example iris class input
#   [0]



# Standardize the features
X = (X - X.mean(axis=0)) / X.std(axis=0)

#print(X)

#sigmoid function used for adjusting output to match the binary classification purpose
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
#sigmoid function derivative used for adjusting weights in backpropogation
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Set hyperparameters
input_neurons = 4 #number of attributes
hidden_neurons = 2  
output_neurons = 1 #representing the class
learning_rate = 0.01
epochs = 1000

#random init of weights to use delta learning rule 
wh = np.random.uniform(size=(input_neurons, hidden_neurons))
bh = np.random.uniform(size=(1, hidden_neurons))
wout = np.random.uniform(size=(hidden_neurons, output_neurons))
bout = np.random.uniform(size=(1, output_neurons))

#print(wh)
#print(bh)
#print(wout)
#print(bout)

# Forward propagation
def forward_propagation(X):
    hidden_layer_input = np.dot(X, wh) + bh
    hidden_layer_activation = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_activation, wout) + bout
    output = sigmoid(output_layer_input)
    return hidden_layer_activation, output

# Backpropagation
def backpropagation(X, y, hidden_layer_activation, output, learning_rate):
    global wh, bh, wout, bout

    # Calculate error

    output_error = y - output
    output_delta = output_error * sigmoid_derivative(output)

    hidden_layer_error = output_delta.dot(wout.T)
    hidden_layer_delta = hidden_layer_error * sigmoid_derivative(hidden_layer_activation)

    # Update weights and biases
    wout += hidden_layer_activation.T.dot(output_delta) * learning_rate
    bout += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
    wh += X.T.dot(hidden_layer_delta) * learning_rate
    bh += np.sum(hidden_layer_delta, axis=0, keepdims=True) * learning_rate

# Train the network
def train(X, y, epochs, learning_rate):
    global wh, bh, wout, bout
    for epoch in range(epochs):
        hidden_layer_activation, output = forward_propagation(X)
        backpropagation(X, y, hidden_layer_activation, output, learning_rate)
        if epoch % 100 == 0:
            loss = np.mean(np.square(y - output))
            print(f'Epoch {epoch}, Loss: {loss}')

# Predict function
def predict(X):
    _, output = forward_propagation(X)
    return output

# Train the network
train(X, y, epochs, learning_rate)

# Predictions
predictions = predict(X)

#print(predictions)

predictions = [1 if p >= 0.5 else 0 for p in predictions]

#print(predictions)

# Evaluate accuracy
accuracy = np.mean(predictions == y.flatten()) * 100
print(f'Accuracy: {accuracy}%')


dataTrain = pd.read_csv('trainingset1.csv')
dataTest = pd.read_csv('testset1.csv')
dataVal = pd.read_csv('validationset1.csv')
dataTrain['Iris Class'] = dataTrain['Iris Class'].map({'Iris-setosa': 0, 'Iris-versicolor': 1})
dataTest['Iris Class'] = dataTest['Iris Class'].map({'Iris-setosa': 0, 'Iris-versicolor': 1})
dataVal['Iris Class'] = dataVal['Iris Class'].map({'Iris-setosa': 0, 'Iris-versicolor': 1})
Xtrain = dataTrain[['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']].values
ytrain = dataTrain['Iris Class'].values.reshape(-1, 1)
Xtest = dataTest[['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']].values
ytest = dataTest['Iris Class'].values.reshape(-1, 1)
Xval = dataVal[['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']].values
yval = dataVal['Iris Class'].values.reshape(-1, 1)