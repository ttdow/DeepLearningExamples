import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):

        # Initalize weights randomly with small values
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.01
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.01
        self.bias2 = np.zeros((1, output_size))

    def forward(self, X):

        # Forward propagation through the network
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = np.tanh(self.z1)
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        y_hat = np.exp(self.z2) / np.sum(np.exp(self.z2), axis=1, keepdims=True)
        return y_hat
    
    def backward(self, X, y, y_hat, learning_rate):

        #Backward propagation through the network
        delta3 = y_hat
        delta3[range(len(X)), y] -= 1
        delta2 = np.dot(delta3, self.weights2.T) * (1 - np.power(self.a1, 2))
        d_weights2 = np.dot(self.a1.T, delta3)
        d_bias2 = np.sum(delta3, axis=0, keepdims=True)
        d_weights1 = np.dot(X.T, delta2)
        d_bias1 = np.sum(delta2, axis=0)

        # Update weights and biases using gradient descent
        self.weights1 -= learning_rate * d_weights1
        self.bias1 -= learning_rate * d_bias1
        self.weights2 -= learning_rate * d_weights2
        self.bias2 -= learning_rate * d_bias2