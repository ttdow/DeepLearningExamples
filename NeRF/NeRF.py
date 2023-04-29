import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

@torch.no_grad()
def test(nerf_model, hn, hf, dataset, chunk_size=10, img_index=0, dir_index=0, nb_bins=192, H=400, W=400, device='cuda'):
    ray_origins = dataset[img_index * H * W: (img_index + 1) * H * W, :3]
    ray_directions = dataset[img_index * H * W: (img_index + 1) * H * W, 3:6]

    data = []
    for i in range(int(np.ceil(H / chunk_size))):
        ray_origins_ = ray_origins[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
        ray_directions_ = ray_directions[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)

        regenerated_px_values = render_rays(nerf_model, ray_origins_, ray_directions_, hn=hn, hf=hf, nb_bins=nb_bins)
        data.append(regenerated_px_values)
    
    img = torch.cat(data).data.cpu().numpy().reshape(H, W, 3)

    plt.figure()
    plt.imshow(img)
    plt.savefig(f'novel_views/{dir_index}-img_{img_index}.png', bbox_inches='tight')
    plt.close()

class NerfModel(nn.Module):
    def __init__(self, embedding_dim_pos=10, embedding_dim_direction=4, hidden_dim=128):
        super(NerfModel, self).__init__()

        self.block1 = nn.Sequential(
            nn.Linear(embedding_dim_pos * 6 + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Linear(embedding_dim_pos * 6 + hidden_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim + 1)
        )

        self.block3 = nn.Sequential(
            nn.Linear(embedding_dim_direction * 6 + hidden_dim + 3, hidden_dim // 2),
            nn.ReLU()
        )

        self.block4 = nn.Sequential(
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid()
        )

        self.embedding_dim_pos = embedding_dim_pos
        self.embedding_dim_direction = embedding_dim_direction
        self.relu = nn.ReLU()

    @staticmethod
    def positional_encoding(x, L):
        out = [x]
        for j in range(L):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))

        return torch.cat(out, dim=1)
    
    def forward(self, o, d):
        emb_x = self.positional_encoding(o, self.embedding_dim_pos)
        emb_d = self.positional_encoding(d, self.embedding_dim_direction)
        h = self.block1(emb_x)
        tmp = self.block2(torch.cat((h, emb_x), dim=1))
        h, sigma = tmp[:, :-1], self.relu(tmp[:, -1])
        h = self.block3(torch.cat((h, emb_d), dim=1))
        c = self.block4(h)

        return c, sigma

def compute_accumulated_transmittance(alphas):
    
    accumulated_transmittance = torch.cumprod(alphas, 1)
    
    return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device), accumulated_transmittance[:, :-1]), dim=-1)

def render_rays(nerf_model, ray_origins, ray_directions, hn=0, hf=0.5, nb_bins=192):
    
    device = ray_origins.device

    # hn = near
    # hf = far

    # Create a set of nb_bins equally-spaced values from hn to hf for all datapoints in ray_origins
    t = torch.linspace(hn, hf, nb_bins, device=device).expand(ray_origins.shape[0], nb_bins)

    # Perturb sampling along each ray.
    # Calculate midpoint of all adjacent bin values
    mid = (t[:, :-1] + t[:, 1:]) / 2.

    # Concatenate all the first bins and all the midpoints
    lower = torch.cat((t[:, :1], mid), -1)

    # Concatenate all the midpoints and all the last bins
    upper = torch.cat((mid, t[:, -1:]), -1)

    # Create random values between [-1, 1] of shape t
    u = torch.rand(t.shape, device=device)

    # Randomly offset each bin value in t
    t = lower + (upper - lower) * u # [batch_size, nb_bins]

    # Find difference between all bin values and concatenate a giant number???
    delta = torch.cat((t[:, 1:] - t[:, :-1], torch.tensor([1e10], device=device).expand(ray_origins.shape[0], 1)), -1)

    # [batch_size, 1, 3] + [batch_size, nb_bins, 1] * [batch_size, 1, 3] = [batch_size, nb_bins, 3] * [batch_size, 1, 3]
    # = [batch_size, nb_bins, 3] = position of all rays at 10 randomly perturbed time intervals along their direction
    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1) # [batch_size, nb_bins, 3]

    # Make a number of ray directions equal to the batch size and the nb_bins
    # i.e. make a corresponding list of ray direction for each point along the rays generated above
    ray_directions = ray_directions.expand(nb_bins, ray_directions.shape[0], 3).transpose(0, 1)

    # Query NN model for color and sigma outputs given the input points along the rays
    colors, sigma = nerf_model(x.reshape(-1, 3), ray_directions.reshape(-1, 3))

    # Reshape the color outputs so they correspond to all the points along the rays
    colors = colors.reshape(x.shape)

    # Reshape the density outputs so they correspond to all the points along the rays
    sigma = sigma.reshape(x.shape[:-1])

    # Calculate alpha values for all points: 1 - e^(-sigma * delta)
    alpha = 1 - torch.exp(-sigma * delta) # [batch_size, nb_bins]

    # Calculate the amount of weight that is transmitted at each point where alpha is proportional
    # to the opacity of the scene at that point
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)

    # Calculate the color of each point given the weight of transmittance and sum each point along a ray
    c = (weights * colors).sum(dim=1) # Pixel values

    # Sum the weights to determine what the maximum weight (i.e. maximum transmittance) would be
    # i.e. no opacity at all
    weight_sum = weights.sum(-1).sum(-1) # Regularization for white background

    return c + 1 - weight_sum.unsqueeze(-1)

def train(nerf_model, optimizer, scheduler, data_loader, device='cuda', hn=0, hf=1, nb_epochs=int(1e5), nb_bins=192, H=400, W=400):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    training_loss = []

    dir_index = 0
    counter = 0
    nb_epochs = 2
    for _ in tqdm(range(nb_epochs)):
        for batch in data_loader:

            print("Batch: " + str(counter) + " / " + str(15625))

            # Batch matrix layout:
            #                ----------------------------------------------------------------------------------------------------------------
            # data_element_0 | ray_origin_x | ray_origin_y | ray_origin_z | ray_direction_x | ray_direction_y | ray_direction_z | r | g | b |
            # data_element_1 | ray_origin_x | ray_origin_y | ray_origin_z | ray_direction_x | ray_direction_y | ray_direction_z | r | g | b |

            # Slice above batch matrix representation into components for computations
            ray_origins = batch[:, :3].to(device)
            ray_directions = batch[:, 3:6].to(device)
            ground_truth_px_values = batch[:, 6:].to(device)

            # Take input ray data and use it to create a light field / 3D scene representation constructed of points
            regenerated_px_values = render_rays(nerf_model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins)

            # Calculate loss of predicted point rgba values and actual rgba values
            loss = ((ground_truth_px_values - regenerated_px_values) ** 2).sum()

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Append loss for logging
            training_loss.append(loss.item())

            counter += 1

        # Advance learning rate scheduler
        scheduler.step()
        counter = 0

        # Construct prediction images of the scene every epoch
        for img_index in range(10):
            test(nerf_model, hn, hf, testing_dataset, img_index=img_index, dir_index=dir_index, nb_bins=nb_bins, H=H, W=W)

        dir_index += 1

    return training_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

training_dataset = torch.from_numpy(np.load('../data/training_data.pkl', allow_pickle=True))
testing_dataset = torch.from_numpy(np.load('../data/testing_data.pkl', allow_pickle=True))

model = NerfModel(hidden_dim=256).to(device)
model_optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=[2, 4, 8], gamma=0.5)

data_loader = DataLoader(training_dataset, batch_size=1024, shuffle=True)
train(model, model_optimizer, scheduler, data_loader, nb_epochs=16, device=device, hn=2, hf=6, nb_bins=192, H=400, W=400)