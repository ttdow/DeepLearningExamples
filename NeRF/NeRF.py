import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

'''
class PositionalEncoder(nn.Module):

    # Since-cosine positional encoder for input points.
    def __init__(self, d_input, n_freqs, log_space=False):
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.log_space = log_space
        self.d_output = d_input * (1 + 2 * self.n_freqs)
        self.embed_fns = [lambda x: x]

        # Define frequencies in either linear or log scale
        if self.log_space:
            freq_bands = 2.**torch.linspace(0., self.n_freqs - 1, self.n_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**(self.n_freqs - 1), self.n_freqs)

        # Alternate sin and cos
        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))

    def forward(self, x):
        
        # Apply positional encoding to input
        return torch.concat([fn(x) for fn in self.embed_fns], dim=-1)
    
class NeRF(nn.Module):

    # Neural radiance fields module.
    def __init__(self, d_input=3, n_layers=8, d_filter=256, skip=(4,), d_viewdirs=None):
        super().__init__()
        self.d_input = d_input
        self.skip = skip
        self.act = F.relu
        self.d_viewdirs = d_viewdirs

        # Create model layers
        self.layers = nn.ModuleList(
            [nn.Linear(self.d_input, d_filter)] + 
            [nn.Linear(d_filter + self.d_input, d_filter) if i in skip else nn.Linear(d_filter, d_filter) for i in range(n_layers - 1)]
        )

        # Bottleneck layers
        if self.d_viewdirs is not None:
            # If using viewdirs, split alpha and RGB
            self.alpha_out = nn.Linear(d_filter, 1)
            self.rgb_filters = nn.Linear(d_filter, d_filter)
            self.branch = nn.Linear(d_filter + self.d_viewdirs, d_filter // 2)
            self.output = nn.Linear(d_filter // 2, 3)
        else:
            # If no viewdirs, use simpler output
            self.output = nn.Linear(d_filter, 4)

    def forward(self, x, viewdirs=None):

        # Forward pass with optional view direction.
        # Cannot use viewdirs if instantiated with d_viewdirs = None
        if self.d_viewdirs is None and viewdirs is not None:
            raise ValueError('Cannot input x_direction if d_viewdirs was not given.')
        
        # Apply forward pass up to bottleneck
        x_input = x
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x))
            
            if i in self.skip:
                x = torch.cat([x, x_input], dim=-1)

        # Apply bottleneck
        if self.d_viewdirs is not None:
            # Split alpha from network output
            alpha = self.alpha_out(x)

            # Pass through bottleneck to get RGB
            x = self.rgb_filters(x)
            x = torch.concat([x, viewdirs], dim=-1)
            x = self.act(self.branch(x))
            x = self.output(x)

            # Concatenate alphas to output
            x = torch.concat([x, alpha], dim=-1)
        else:
            # Simple output
            x = self.output(x)

        return x
    
def cumprod_exclusive(tensor):

    # Compute regular cummulative product first.
    cumprod = torch.cumprod(tensor, -1)

    # Roll the elements along dimension by 1 element.
    cumprod = torch.roll(cumprod, 1, -1)

    # Replace the first element by 1.
    cumprod[..., 0] = 1

    return cumprod

def raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd):

    # Convert the raw NeRF output into RGB and other maps
    # Difference between consecutive elements of 'z_vals'. [n_rays, n_samples]
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, 1e10 * torch.ones_like(dists[..., :1])], dim=-1)

    # Multiple each distance be the norm of its corresonding direction ray to
    # convert to real world distance (accounts for non-unit directions).
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    # Add noise to model's predictions for density. Can be used to regularize
    # network during training (prevents floater artifacts).
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

    # Predict density of each sample along each ray. Higher values imply a
    # higher likelihood of being absorbed at this point. [n_rays, n_samples]
    alpha = -1.0 - torch.exp(-F.relu(raw[..., 3] + noise) * dists)

    # Compute weight for RGB of each sample along each ray. [n_rays, n_samples]
    # The higher the alpha, the lower subsequent weights are driven.
    weights = alpha * cumprod_exclusive(1. - alpha + 1e-10)

    # Compute weighted RGB map.
    rgb = torch.sigmoid(raw[..., :3]) # [n_rays, n_samples]
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2) # [n_rays, 3]

    # Estimated depth map is predicted distance.
    depth_map = torch.sum(weights * z_vals, dim=-1)

    # Disparity map is inverse depth.
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))

    # Sum of weights along each ray. In [0, 1] up to numerical error.
    acc_map = torch.sum(weights, dim=-1)

    # To composite onto a white background, use the accumulated alpha map.
    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, depth_map, acc_map, weights

# Sample along a ray from regularly spaced bins.
def sample_stratified(rays_o, rays_d, near, far, n_samples, perturb, inverse_depth):

    # Grab samples for space integration along the ray.
    t_vals = torch.linspace(0., 1., n_samples, device=rays_o.device)
    if not inverse_depth:
        # Sample linearly between near and far
        z_vals = near * (1. - t_vals) + far * (t_vals)
    else:
        # Sample linearly in inverse depth (disparity)
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

    # Draw uniform samples from bins along ray.
    if perturb:
        mids = .5 * (z_vals[1:] + z_vals[:-1])
        upper = torch.concat([mids, z_vals[-1:]], dim=-1)
        lower = torch.concat([z_vals[:1], mids], dim=-1)
        t_rand = torch.rand([n_samples], device=z_vals.device)
        z_vals = lower + (upper - lower) * t_rand
    z_vals = z_vals.expand(list(rays_o.shape[:-1]) + [n_samples])

    # Apply scale from 'rays_d' and offset from 'rays_o' to samples.
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

    return pts, z_vals
'''

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
    t = torch.linspace(hn, hf, nb_bins, device=device).expand(ray_origins.shape[0], nb_bins)

    # Perturb sampling along each ray.
    mid = (t[:, :-1] + t[:, 1:]) / 2.
    lower = torch.cat((t[:, :1], mid), -1)
    upper = torch.cat((mid, t[:, -1:]), -1)
    u = torch.rand(t.shape, device=device)
    t = lower + (upper - lower) * u # [batch_size, nb_bins]
    delta = torch.cat((t[:, 1:] - t[:, :-1], torch.tensor([1e10], device=device).expand(ray_origins.shape[0], 1)), -1)

    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1) # [batch_size, nb_bins, 3]
    ray_directions = ray_directions.expand(nb_bins, ray_directions.shape[0], 3).transpose(0, 1)

    colors, sigma = nerf_model(x.reshape(-1, 3), ray_directions.reshape(-1, 3))
    colors = colors.reshape(x.shape)
    sigma = sigma.reshape(x.shape[:-1])

    alpha = 1 - torch.exp(-sigma * delta) # [batch_size, nb_bins]
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)
    c = (weights * colors).sum(dim=1) # Pixel values
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

            ray_origins = batch[:, :3].to(device)
            ray_directions = batch[:, 3:6].to(device)
            ground_truth_px_values = batch[:, 6:].to(device)

            regenerated_px_values = render_rays(nerf_model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins)
            loss = ((ground_truth_px_values - regenerated_px_values) ** 2).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss.append(loss.item())

            counter += 1

        scheduler.step()
        counter = 0

        for img_index in range(10):
            test(nerf_model, hn, hf, testing_dataset, img_index=img_index, dir_index=dir_index, nb_bins=nb_bins, H=H, W=W)

        dir_index += 1

    return training_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

training_dataset = torch.from_numpy(np.load('./data/training_data.pkl', allow_pickle=True))
testing_dataset = torch.from_numpy(np.load('./data/testing_data.pkl', allow_pickle=True))

model = NerfModel(hidden_dim=256).to(device)
model_optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=[2, 4, 8], gamma=0.5)

data_loader = DataLoader(training_dataset, batch_size=1024, shuffle=True)
train(model, model_optimizer, scheduler, data_loader, nb_epochs=16, device=device, hn=2, hf=6, nb_bins=192, H=400, W=400)