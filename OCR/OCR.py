import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        x = self.fc(h.squeeze(0))
        return x
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_epochs = 10
batch_size = 64
lr = 0.001

im0 = Image.open("test.jpg")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

img = transform(im0)

label = 

input_size = 100
hidden_size = 128
num_classes = 26

model = Model(input_size, hidden_size, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(num_epochs):
    
    # Forward pass.
    output = model(img)
    print(output)
    loss = criterion(output, label)

    # Backward pass.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Epoch: " + str(epoch) + ", Loss: " + str(loss))

torch.save(model.state_dict(), "model.pt")