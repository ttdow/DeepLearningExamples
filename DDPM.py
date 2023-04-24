import random
import numpy as np
import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.transforms import Compose, ToTensor, Lambda
from torchvision.datasets.mnist import MNIST, FashionMNIST

def show_images(images, title=""):

    # Converting images to CPU numpy arrays
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()

    # Defining number of rows and columns
    fig = plt.figure(figsize=(8,8))
    rows = int(len(images) ** (1/2))
    cols = round(len(images) / rows)

    # Populating figure with subplots
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx+1)

            if idx < len(images):
                plt.imshow(images[idx][0], cmap='gray')
                idx += 1
    
    fig.suptitle(title, fontsize=30)

    # Display the figure
    plt.show()

class DDPM(nn.Module):
    def __init__(self, network, n_steps=200, min_beta=10**-4, max_beta=0.02, device=torch.device("cpu"), image_ch=(1, 28, 28)):
        super(DDPM, self).__init__()
        
        self.n_steps = n_steps
        self.device = device
        self.image_ch = image_ch
        self.network = network.to(self.device)
        self.betas = torch.linspace(min_beta, max_beta, self.n_steps).to(self.device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i+1]) for i in range(len(self.alphas))]).to(self.device)

    def forward(self, x0, t, eta=None):
        
        # Make input image more noisy (we can skip directly to the desired step)
        n, c, h, w = x0.shape
        a_bar = self.alpha_bars[t]

        if eta is None:
            eta = torch.randn(n, c, h, w).to(self.device)

        noisy = a_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eta

        return noisy

    def backward(self, x, t):

        # Run each image through the network for each timestep t in the vector t.
        # The network returns its estimation of the noise that was added.
        return self.network(x, t)


SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MNIST_PATH = f"ddpm_model_mnist.pt"
FASHION_PATH = f"ddpm_model_fashion.pt"

# Hyperparameters
batch_size = 128
n_epochs = 20
lr = 0.001

# Transforms for images in dataset (convert to tensor and normalize)
transform = Compose([
    ToTensor(),
    Lambda(lambda x: (x - 0.5) * 2)
])

# Download dataset
mnist_data = MNIST("./data", download=True, train=True, transform=transform)

# Prepare data loader
data = DataLoader(mnist_data, batch_size, shuffle=True)

show_images(next(iter(data))[0])


'''
test = next(iter(data))

print(type(test))
print(len(test))
print(type(test[0]))

transform = T.ToPILImage()
img = transform(test[0][0])
img.show()
'''