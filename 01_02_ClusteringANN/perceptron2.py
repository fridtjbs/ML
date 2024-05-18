import numpy as np
import pandas as pd



# Load dataset
data = pd.read_csv('irisBinary.csv')


data['Iris Class'] = data['Iris Class'].map({'Iris-setosa': 0, 'Iris-versicolor': 1})

# Split dataset into features and labels
X = data[['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']].values
y = data['Iris Class'].values.reshape(-1, 1)


#imported library randomly split testset, not related to the perceptron itself, source: https://www.analyticsvidhya.com/blog/2023/11/train-test-validation-split/
from sklearn.model_selection import train_test_split
Xtrain, Xtemp, ytrain, ytemp = train_test_split(X, y, test_size=0.4, random_state=42)
Xval, Xtest, yval, ytest = train_test_split(Xtemp, ytemp, test_size=0.5, random_state=42)
#import csv

print(Xtest)
print(ytest)




# Map class labels to binary values
data['Iris Class'] = data['Iris Class'].map({'Iris-setosa': 0, 'Iris-versicolor': 1})








#   example iris attribute input
#   [5.1 3.5 1.4 0.2]

#   example iris class input
#   [0]



# Standardize the features
#X = (X - X.mean(axis=0)) / X.std(axis=0)

#print(X)

#sigmoid function used for adjusting output to match the binary classification purpose
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
#sigmoid function derivative used for adjusting weights in backpropogation
def sigmoid_derivative(x):
    return x * (1 - x)


# Set hyperparameters
input_neurons = 4 #number of attributes

output_neurons = 1 #representing the class
learning_rate = 0.01
epochs = 1000

#random init of weights to use delta learning rule 
weight = np.random.uniform(size=(input_neurons, output_neurons))
bias = np.random.uniform(size=(1, output_neurons))

#print(wh)
#print(bh)
#print(wout)
#print(bout)

# Forward propagation
def forward_propagation(X):
    output_layer_input = np.dot(X, weight) + bias
    output = sigmoid(output_layer_input)
    return  output



# Train the network
def train(X, y, epochs, learning_rate):
    global weight, bias
    for epoch in range(epochs):
        output = forward_propagation(X)
        
        error = y - output
        delta  = error * sigmoid_derivative(output)

        weight += np.dot(X.T, delta) * learning_rate
        bias += np.sum(delta, axis=0, keepdims=True) * learning_rate

        if epoch % 100 == 0:
            loss = np.mean(np.square(y - output))
            print(f'Epoch {epoch}, Loss: {loss}')

# Predict function
def predict(X):
    output = forward_propagation(X)
    return output

# Train the network
train(Xtrain, ytrain, epochs, learning_rate)

# Predictions
predictions = predict(Xtest)

#print(predictions)
predictions = [1 if p >= 0.5 else 0 for p in predictions]

print(predictions)

# Evaluate accuracy
accuracy = np.mean(predictions == ytest.flatten()) * 100
print(f'Accuracy: {accuracy}%')

