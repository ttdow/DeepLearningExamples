
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torchvision
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.onnx

# Function to convert to ONNX.
def Convert_ONNX(model):

	# Set the model to inference model.
	model.eval()

	input_size = (1, 3, 32, 32)

	# Let's create a dummy input tensor.
	dummy_input = torch.randn(input_size, requires_grad=True)

	# Export the model
	torch.onnx.export(model, 
				      dummy_input, 
					  "ImageClassifier.onnx", 
					  export_params=True, 
					  opset_version=10, 
					  do_constant_folding=True, 
					  input_names=['modelInput'], 
					  output_names=['modelOutput'], 
					  dynamic_axes={'modelInput' : {0 : 'batch_size'}, 
					                'modelOutput' : {0 : 'batch_size'}})

	# Print result.
	print("\n\nModel has been converted to ONNX format.")

# Define a convolutional neural network.
class Network(nn.Module):
	def __init__(self):
		super(Network, self).__init__()

		self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1)
		self.bn1 = nn.BatchNorm2d(12)
		self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(12)
		self.pool = nn.MaxPool2d(2,2)
		self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
		self.bn4 = nn.BatchNorm2d(24)
		self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
		self.bn5 = nn.BatchNorm2d(24)
		self.fc1 = nn.Linear(24*10*10, 10)

	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))      
		x = F.relu(self.bn2(self.conv2(x)))     
		x = self.pool(x)                        
		x = F.relu(self.bn4(self.conv4(x)))     
		x = F.relu(self.bn5(self.conv5(x)))     
		x = x.view(-1, 24*10*10)
		output = self.fc1(x)

		return output

# Function to save the model.
def SaveModel(model):
	path = "./model.pth"
	torch.save(model.state_dict(), path)

# Function to test the model with the test dataset and print the accuracy for the test images.
def TestAccuracy(model, test_loader):

	model.eval()
	accuracy = 0.0
	total = 0.0

	with torch.no_grad():
		for data in test_loader:
			images, labels = data

			# Run the model on the test set to predict labels.
			outputs = model(images)

			# The label with the highest energy will be our prediction.
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			accuracy += (predicted == labels).sum().item()

		# Compute the accuracy over all test images.
		accuracy = (100 * accuracy / total)
		return(accuracy)

# Training function. We simply have to loop over our data iterator and feed the inputs to the network and optimize.
def Train(num_epochs, model=None):

	# Loading and normalizing the data.
	# Define the transformations for the training and test sets.
	transformations = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

	# CIFAR10 dataset consists of 50K training images. We define the batch size of 10 to load 5000 batches of images.
	batch_size = 10
	number_of_labels = 10

	# Create an instance for training.
	# When we run this code for the first time, the CIFAR10 dataset will be downloaded locally.
	train_set = CIFAR10(root="./data", train=True, transform=transformations, download=True)

	# Create a loader for the training set which will read the data within batch_size and put into memory.
	train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
	print("The number of images in a training set is: ", len(train_loader) * batch_size)

	# Create an instance for testing, note that train is set to false.
	# When we run this code for the first time, the CIFAR10 test dataset will be downloaded locally.
	test_set = CIFAR10(root="./data", train=False, transform=transformations, download=True)

	# Create a loader for the testing set which will read the data within batch_size and put into memory.
	# Note that each shuffle is set to false for the test loader.
	test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
	print("The number of images in a testing set is: ", len(test_loader) * batch_size)

	print("The number of batches per episode is: ", len(train_loader))
	classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	# Instantiate a neural network model.
	if model == None:
		model = Network()

	# Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer.
	loss_fn = nn.CrossEntropyLoss()
	optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

	best_accuracy = 0.0

	# Define your execution device.
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print("The model will be running on", device, "device")

	# Convert model parameters and buffers to CPU or Cuda.
	model.to(device)

	# Loop over dataset multiple times.
	for epoch in range(num_epochs):
		running_loss = 0.0
		running_acc = 0.0

		for i, (images, labels) in enumerate(train_loader, 0):

			# Get the inputs.
			images = Variable(images.to(device))
			labels = Variable(labels.to(device))

			# Zero the parameter gradients.
			optimizer.zero_grad()

			# Predict the classes using images from the training set.
			outputs = model(images)

			# Compute the loss based on model output and real labels.
			loss = loss_fn(outputs, labels)

			# Backpropagate the loss.
			loss.backward()

			# Adjust the parameters based on the calculated gradients.
			optimizer.step()

			# Extract the loss value.
			running_loss += loss.item()

			if i % 1000 == 999:

				# Print every 1000 (twice per epoch).
				print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))

				# Zero the loss.
				running_loss = 0.0

		# Compute and print the average accuracy of this epoch when tested over all 10000 test images.
		accuracy = TestAccuracy(model, test_loader)
		print('For epoch', epoch+1, 'the test accuracy over the whole test set is %d %%' % (accuracy))

		# We want to save the model if the accuracy is the best.
		if accuracy > best_accuracy:
			SaveModel(model)
			best_accuracy = accuracy

	return model, test_loader

# Function to show the images.
def ImageShow(img):
	img = img / 2 + 0.5
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()

# Function to test the model with a batch of images and show the labels predictions.
def TestBatch(model, test_loader):

	# Get batch of images from the test DataLoader.
	images, labels = next(iter(test_loader))

	# Show all the images as one image grid.
	ImageShow(torchvision.utils.make_grid(images))

	batch_size = 10
	classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	# Show the real labels on the screen.
	print('Real labels: ', ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

	# Let's see what the model identifies the images as.
	outputs = model(images)

	# We got the probability for every 10 labels. The highest (max) probability should be correct label
	_, predicted = torch.max(outputs, 1)

	# Let's show the predicted labels on the screen to compare with the real ones
	print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(batch_size)))

if __name__ == '__main__':

	# Load model from checkpoint.
	#model = Network()
	#path = "model.pth"
	#model.load_state_dict(torch.load(path))

	# Build the model.
	#model, test_loader = Train(1, model)
	#print('Finished training.')

	# Test which classes performed well.
	#TestAccuracy(model, test_loader)

	# Let's load the model we just created and test the accuracy per label.
	model = Network()
	path = "model.pth"
	model.load_state_dict(torch.load(path))

	# Test with batch of images.
	#TestBatch(model, test_loader)

	# Convert to ONNX.
	Convert_ONNX(model)