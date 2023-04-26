from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage
from torchvision.datasets.mnist import MNIST

class Text2Image(nn.Module):
    def __init__(self, text_embedding_size, image_size, batch_size):
        super(Text2Image, self).__init__()

        self.text_embedding_size = text_embedding_size
        self.image_size = image_size
        self.batch_size = batch_size

        self.text_encoder = nn.Sequential(
            nn.Linear(self.text_embedding_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU()
        )

        self.image_decoder = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.image_size**2),
            nn.Tanh()
        )

    def forward(self, text_embedding):

        text_embedding = torch.unsqueeze(text_embedding, 1).float()

        encoded_text = self.text_encoder(text_embedding)
        generated_image = self.image_decoder(encoded_text)

        return generated_image.reshape(self.batch_size, self.image_size, self.image_size)
    
SEED = 0
#random.seed(SEED)
#np.random.seed(SEED)
#torch.manual_seed(SEED)

transform = Compose([
    ToTensor(),
    Lambda(lambda x: (x - 0.5) * 2)
])

batch_size = 512

mnist_data = MNIST("./data", download=True, train=True, transform=transform)

dataloader = DataLoader(mnist_data, batch_size=batch_size, shuffle=True)

generator = Text2Image(text_embedding_size=1, image_size=28, batch_size=batch_size)
generator.load_state_dict(torch.load('checkpoint.pt'))

criterion = nn.MSELoss()
optimizer = optim.Adam(generator.parameters(), lr=0.0002)

x_test = torch.randint(10, (batch_size,))
test_output = generator(x_test)
print(x_test[0])

transform = ToPILImage()
img = transform(test_output[0])
img.show()



counter = 0
for epoch in range(100):
    for target_images, text_embeddings in dataloader:

        if len(text_embeddings) < batch_size:
            continue

        #print("Batch: " + str(counter) + " / " + str(len(dataloader)))
        counter += 1

        # Create images from text
        generated_images = generator(text_embeddings)

        # Calculate loss as compared to target image
        loss = criterion(generated_images, target_images.squeeze(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")
    torch.save(generator.state_dict(), "checkpoint.pt")

x_test = torch.randint(10, (batch_size,))
test_output = generator(x_test)

transform = ToPILImage()
img = transform(test_output[0])
img.show()