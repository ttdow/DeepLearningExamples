import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Adam
from torch.utils.data import random_split, DataLoader, TensorDataset
from torch.autograd import Variable

# Define neural network.
class Network(nn.Module):
    def __init__(self, input_size, output_size):
        super(Network, self).__init__()

        self.layer1 = nn.Linear(input_size, 24)
        self.layer2 = nn.Linear(24, 24)
        self.layer3 = nn.Linear(24, output_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        y = self.layer3(x)

        return y

def SaveModel(model):
    path = "./NetModel.pth"
    torch.save(model.state_dict(), path)

def Train(num_epochs, model, train_loader, validate_loader):

    # Define the loss function and optimizer.
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=0.0001)

    best_accuracy = 0.0

    for epoch in range(1, num_epochs+1):
        running_train_loss = 0.0
        running_accuracy = 0.0
        running_val_loss = 0.0
        total = 0.0

        # Training loop.
        for data in train_loader:
            inputs, outputs = data
            optimizer.zero_grad()
            predicted_outputs = model(inputs).float()
            _, indices = torch.max(predicted_outputs, dim=1)

            train_loss = loss_fn(indices.float(), outputs.float())
            train_loss.backward()
            optimizer.step()
            running_train_loss += train_loss.item()

        # Calculate the training loss value.
        train_loss_value = running_train_loss / len(train_loader)

        # Validation loop.
        with torch.no_grad():
            model.eval()
            for data in validate_loader:
                inputs, outputs = data
                predicted_outputs = model(inputs)
                _, indices = torch.max(predicted_outputs, dim=1)
                val_loss = loss_fn(indices, outputs)

                # The label with the highest value will be our prediction.
                _, predicted = torch.max(predicted_outputs, 1)
                running_val_loss += val_loss.item()
                total += outputs.size(0)
                running_accuracy += (predicted == outputs).sum().item()

        # Calculate the validation loss.
        val_loss_value = running_val_loss / len(validate_loader)

        # Calculate accuracy as the number of correct predictions in the validation batch divided
        # by the total number of predictions done.
        accuracy = (100 * running_accuracy / total)

        # Save the model if the accuracy is the best.
        if accuracy > best_accuracy:
            SaveModel(model)
            best_accuracy = accuracy

        # Print the statistics after each epoch.
        if epoch % 10 == 0:
            print('Epoch:', epoch, 'Training Loss:', train_loss_value, 'Validation Loss:', val_loss_value, 'Accuracy:', accuracy)

def Test(input_size, output_size, test_loader):

    # Load the best model.
    model = Network(input_size, output_size)
    path = "./NetModel.pth"
    model.load_state_dict(torch.load(path))

    running_accuracy = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            inputs, outputs = data
            outputs = outputs.to(torch.float32)
            predicted_outputs = model(inputs)
            _, predicted = torch.max(predicted_outputs, 1)
            total += outputs.size(0)
            running_accuracy += (predicted == outputs).sum().item()

        print('Test Accuracy:', (100 * running_accuracy / total))

# Loading the data.
df = pd.read_excel(r'./Iris_dataset.xlsx')
print(df.head())

# Let's verify if our data is balanced and what types of species we have.
print(df['Iris_Type'].value_counts())

# Convert Iris species into numeric types: Iris-setosa=0, Iris-versicolor=1, Iris-virginica=2.
labels = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
df['IrisType_num'] = df['Iris_Type']
df.IrisType_num = [labels[item] for item in df.IrisType_num]

# Define input and and output datasets.
input = df.iloc[:, 1:-2]
output = df.loc[:, 'IrisType_num']

print('\nInput values:')
print(input.head())

print('\nOutput values:')
print(output.head())

# Convert the input and output data to tensors and create a TensorDataset.
input = torch.Tensor(input.to_numpy())
print('\nInput format: ', input.shape, input.dtype)
output = torch.Tensor(output.to_numpy())
print('\nOutput format: ', output.shape, output.dtype)
data = TensorDataset(input, output)

# Split to train, validate, and test datasets using random_split.
train_batch_size = 10
number_rows = len(input)
test_split = int(number_rows * 0.3)
validate_split = int(number_rows * 0.2)
train_split = number_rows - test_split - validate_split
train_set, validate_set, test_set = random_split(data, [train_split, validate_split, test_split])

# Create a DataLoader to read the data within batch sizes and put into memory.
train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
validate_loader = DataLoader(validate_set, batch_size=1)
test_loader = DataLoader(test_set, batch_size=1)

# Define model parameters.
input_size = list(input.shape)[1]
learning_rate = 0.01
output_size = len(labels)

model = Network(input_size, output_size)

# Define your execution device.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)
model.to(device)

num_epochs = 1000
Train(num_epochs, model, train_loader, validate_loader)

print('Finished training.')

Test(input_size, output_size, test_loader)

print('Finished testing.')