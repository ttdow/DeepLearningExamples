import numpy as np
import struct
from array import array
from os.path import join
import random
import matplotlib.pyplot as plt

class MNISTDataLoader(object):
    def __init__(self, train_image_path, train_label_path, test_image_path, test_label_path):
        self.train_image_path = train_image_path
        self.train_label_path = train_label_path
        self.test_image_path = test_image_path
        self.test_label_path = test_label_path

    def read_image_labels(self, image_path, label_path):

        labels = []

        with open(label_path, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError
            labels = array("B", file.read())

        with open(image_path, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError
            image_data = array("B", file.read())

        images = []

        for i in range(size):
            images.append([0] * rows * cols)

        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_image_labels(self.train_image_path, self.train_label_path)
        x_test, y_test = self.read_image_labels(self.test_image_path, self.test_label_path)

        return (x_train, y_train), (x_test, y_test)


# Parent class for the neural network layers.
class Layer():
    def __init__(self):
        self.params = {}

    def forward(self, inputs):
        raise NotImplementedError
    
    def backward(self, grad):
        raise NotImplementedError
    
    def update_params(self, learning_rate):
        pass

# A fully-connected neural network layer.
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
    
def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(30,20))
    index = 1    
    for x in zip(images, title_texts):        
        image = x[0]        
        title_text = x[1]
        plt.subplot(rows, cols, index)        
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 15);        
        index += 1

    plt.show()

input_path = './data/archive'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

mnist_dataloader = MNISTDataLoader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

images_2_show = []
titles_2_show = []
for i in range(0, 10):
    r = random.randint(1, 60000)
    images_2_show.append(x_train[r])
    titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))    

for i in range(0, 5):
    r = random.randint(1, 10000)
    images_2_show.append(x_test[r])        
    titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))    

show_images(images_2_show, titles_2_show)

# Instantiate neural network.
# All this does is create a blank list of neural network layers.
network = NeuralNetwork()

# Add a linear layer to the neural network.
# Create two np arrays:
#   1. Size [input_size, output_size] representing NN weights.
#   2. Size [output_size] representing NN biases.
network.add(Linear(input_size=784, output_size=128))

# Add a rectilinear activtion unit layer to the neural network.
network.add(ReLU())
network.add(Linear(input_size=128, output_size=128))
network.add(ReLU())
network.add(Linear(input_size=128, output_size=10))

# Add a softmax activation layer to the neural network.
network.add(Softmax())

# Train the neural network.
network.train(inputs, targets, learning_rate=0.001, num_epochs=100)