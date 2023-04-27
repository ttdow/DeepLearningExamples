from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, Grayscale

# Text-to-Image
# 1. Encode text
# 2. Decode to multi-view images

class Text2Image(nn.Module):
    def __init__(self, text_embedding_size, image_size, batch_size):
        super(Text2Image, self).__init__()

        self.text_embedding_size = text_embedding_size
        self.image_width = image_size[0]
        self.image_height = image_size[1]
        self.batch_size = batch_size

        self.text_encoder = nn.Sequential(
            nn.Linear(self.text_embedding_size, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU()
        )

        self.image_decoder = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, self.image_width*self.image_height),
            nn.Tanh()
        )

    def forward(self, x, batch_size):

        encoded_text = self.text_encoder(x.float())
        generated_image = self.image_decoder(encoded_text)

        return generated_image.reshape(batch_size, self.image_width, self.image_height)

generator = Text2Image(text_embedding_size=1, image_size=(100, 100), batch_size=3)
generator.load_state_dict(torch.load('checkpoint.pt'))

criterion = nn.MSELoss()
optimizer = optim.Adam(generator.parameters(), lr=0.0002)

front = Image.open("./giraffe/giraffe_front(small).png").convert('L')
side = Image.open("./giraffe/giraffe_side(small).png").convert('L')
back = Image.open("./giraffe/giraffe_back(small).png").convert('L')

transform = Compose([
    ToTensor()
])

front = transform(front)
side = transform(side)
back = transform(back)

target_images = torch.stack((front, side, back), 1).squeeze(0)
target_images = target_images.permute(0, 2, 1)

#print(target_images.shape)
#transform = ToPILImage()
#img = transform(front)
#img.show()

f = torch.tensor(ord('f'), dtype=torch.int64)
s = torch.tensor(ord('s'), dtype=torch.int64)
b = torch.tensor(ord('b'), dtype=torch.int64)
text = torch.stack((f, s, b), 0).unsqueeze(1)

print(text.shape)

best_loss = float("inf")
for epoch in range(1000):

    # Generate images from text
    generated_images = generator(text, 3)

    # Calculate loss as compared to target image
    loss = criterion(generated_images, target_images)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")
    if loss.item() <= best_loss:
        best_loss = loss.item()
        torch.save(generator.state_dict(), "checkpoint.pt")

x = torch.tensor(ord('b'), dtype=torch.int64).unsqueeze(-1)

test_output = generator(x, 1)

transform = ToPILImage()
img = transform(test_output)
img.show()
img.save("output_s.jpg")

# Images-to-Model
# 2. Ray casting
# 3. NN training
# 4. Voxel grid creation
# 5. Surface mesh creation
# 6. Output to <filetype>