import numpy as np

class Layer():
    def __init__(self):
        self.paramas = {}

    def forward(self, inputs):
        raise NotImplementedError
    
    def backward(self, grad):
        raise NotImplementedError
    
    def update_params(self, learning_rate):
        pass

class Linear(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.params['weights'] = np.random.randn(input_size, output_size)
        self.params['bias'] = np.random.randn(output_size)

    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(inputs, self.params['weights']) + self.params['bias']
    
    def backward(self, grad):
        self.grad_weights = np.dot(self.inputs.T, grad)
        self.grad_bias = np.sum(grad, axis=0)

        return np.dot(grad, self.params['weights'].T)
    
    def update_params(self, learning_rate):
        self.params['weights'] -= learning_rate * self.grad_weights
        self.params['bias'] -= learning_rate * self.grad_bias

class ReLU(Layer):
        def forward(self, inputs):
            self.inputs = inputs

            return np.maximum(0, inputs)
        
        def backward(self, grad):
            return grad * (self.inputs > 0)
        
class Softmax(Layer):
    def forward(self, inputs):
        exps = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.probs = exps / np.sum(exps, axis=1, keepdims=True)

        return self.probs
    
    def backward(self, grad):
        return grad * self.probs * (1 - self.probs)
    
class NeuralNetwork():
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)

        return inputs
    
    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update_params(self, learning_rate):
        for layer in self.layers:
            layer.update_params(learning_rate)

    def train(self, inputs, targets, learning_rate, num_epochs):
        for epoch in range(num_epochs):
            outputs = self.forward(inputs)
            loss = self.compute_loss(outputs, targets)
            accuracy = self.compute_accuracy(outputs, targets)

            self.backward(self.compute_grad(outputs, targets))
            self.update_params(learning_rate)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    def compute_loss(self, outputs, targets):
        return -np.mean(targets * np.log(outputs + 1e-10))

    def compute_grad(self, outputs, targets):
        return (outputs - targets) / outputs.shape[0]

    def compute_accuracy(self, outputs, targets):
        prediction = np.argmax(outputs, axis=1)
        labels = np.argmax(outputs, axis=1)

        return np.mean(prediction == labels)
    
network = NeuralNetwork()
network.add(Linear(input_size=784, output_size=128))
network.add(ReLU())
network.add(Linear(input_size=128, output_size=128))
network.add(ReLU())
network.add(Linear(input_size=128, output_size=10))
network.add(Softmax())