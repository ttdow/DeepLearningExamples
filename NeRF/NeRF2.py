import os
from typing import Optional, Tuple, List, Union, Callable
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from tqdm import trange

import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

# --------------------------- POSITIONAL ENCODER ------------------------------
# Sine-cosine positional encoder for input points.
class PositionalEncoder(nn.Module):
    
    def __init__(self, d_input: int, n_freqs: int, log_space:bool = False):
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.log_space = log_space
        self.d_output = d_input * (1 + 2 * self.n_freqs)
        self.embed_fns = [lambda x: x]

        #Define frequencies in either linear or log scale.
        if self.log_space:
            freq_bands = 2. ** torch.linspace(0., self.n_freqs - 1, self.n_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** (self.n_freqs - 1), self.n_freqs)

        # Alternate sin and cos functions.
        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))

    # Apply positional encoding to input.
    def forward(self, x) -> torch.Tensor:
        return torch.concat([fn(x) for fn in self.embed_fns], dim=-1)

# ----------------------------- NEURAL NETWORK --------------------------------
# Neural radiance fields module.
class NeRF(nn.Module):
    def __init__(self, d_input: int = 3, n_layers: int = 8, d_filter: int = 256, skip: Tuple[int] = (4,), d_viewdirs: Optional[int] = None):
        super().__init__()
        self.d_input = d_input
        self.skip = skip
        self.act = F.relu
        self.d_viewdirs = d_viewdirs

        # Create model layers.
        self.layers = nn.ModuleList(
            [nn.Linear(self.d_input, d_filter)] +
            [nn.Linear(d_filter + self.d_input, d_filter) if i in skip else nn.Linear(d_filter, d_filter) for i in range(n_layers - 1)]
        )

        # Bottleneck layers.
        # If using view directions, split alpha and RGB
        if self.d_viewdirs is not None:
            self.alpha_out = nn.Linear(d_filter, 1)
            self.rgb_filters = nn.Linear(d_filter, d_filter)
            self.branch = nn.Linear(d_filter + self.d_viewdirs, d_filter // 2)
            self.output = nn.Linear(d_filter // 2, 3)

        # If no view directions, use simpler output
        else:
            self.output = nn.Linear(d_filter, 4)

    # Forward pass with optional view direction.
    def forward(self, x: torch.Tensor, viewdirs: Optional[torch.Tensor] = None, show: bool = False) -> torch.Tensor:

        # Cannot use view directions if instantiated with d_viewdirs = None
        if self.d_viewdirs is None and viewdirs is not None:
            raise ValueError('Cannot input x_direction if d_viewdirs was not given.')
        
        # Apply forward pass up to bottlneck.
        x_input = x
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x))
            if i in self.skip:
                x = torch.cat([x, x_input], dim=-1)

        # Apply bottleneck.
        if self.d_viewdirs is not None:

            # Split alpha from network output.
            alpha = self.alpha_out(x)

            # Pass through bottleneck to get RGB.
            x = self.rgb_filters(x)
            x = torch.concat([x, viewdirs], dim=-1)
            x = self.act(self.branch(x))
            x = self.output(x)

            # Concatenate alphas to output.
            x = torch.concat([x, alpha], dim=-1)

        # Simple output.
        else:
            x = self.output(x)

        if show:
            temp = x.reshape(4, -1).detach().numpy()
            fig = plt.figure(figsize=(100,100))
            ax = plt.axes(projection="3d")
            ax.scatter3D(temp[0], temp[1], temp[2], color='green')
            plt.show()

        return x
    
# --------------------------- VOLUME RENDERING --------------------------------
# Compute the exclusive cumulative product.
def cumprod_exclusive(tensor: torch.Tensor) -> torch.Tensor:

    # Compute regular cumulative product first.
    cumprod = torch.cumprod(tensor, -1)

    # 'Roll' the elements along dimension 'dim' by 1 element.
    cumprod = torch.roll(cumprod, 1, -1)

    # Replace the first element by '1'.
    cumprod[..., 0] = 1

    return cumprod

# Convert raw NeRF output into RGB and other maps.
def raw2outputs(raw: torch.Tensor, 
                z_vals: torch.Tensor, 
                rays_d: torch.Tensor, 
                raw_noise_std: float = 0.0, 
                white_bkgd: bool = False
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    
    #print(z_vals.shape)
    #print(z_vals[0])
    #print(rays_d.shape)
    
    # Difference between consecutive elements of 'z_vals'. [n_rays, n_samples]
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, 1e10 * torch.ones_like(dists[..., :1])], dim=-1)

    # Multiply each distance by the norm of its corresponding direction ray to convert to real world distance (accounts for non-unit directions).
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    # Add noise to model's predictions for density. Can be used to regularize network during training (prevents floter artificats).
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std
    
    # Predict density of each sample along each ray. Higher values imply higher likelihood of being absorbed at this point. [n_rays, n_samples]
    alpha = 1.0 - torch.exp(-F.relu(raw[..., 3] + noise) * dists)

    # Compute weight for RGB of each sample along each ray. [n_rays, n_samples]
    # The higher the alpha, the lower subsequent weights are driven.
    weights = alpha * cumprod_exclusive(1. - alpha + 1e-10)

    # Compute weighted RGB map.
    rgb = torch.sigmoid(raw[..., :3]) # [n_rays, n_samples, 3]
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

# Apply inverse transform sampling to a weighted set of points.
def sample_pdf(bins: torch.Tensor,
               weights: torch.Tensor,
               n_samples: int,
               perturb: bool = False
              ) -> torch.Tensor:
    
    # Normalize weights to get PDF.
    pdf = (weights + 1e-5) / torch.sum(weights + 1e-5, -1, keepdims=True) # [n_rays, weights.shape[-1]]

    # Convert PDF to CDF.
    cdf = torch.cumsum(pdf, dim=-1) # [n_rays, weights.shape[-1]]
    cdf = torch.concat([torch.zeros_like(cdf[..., 1]), cdf], dim=-1) # [n_rays, weights.shape[-1] + 1]

    # Take sample positions to grab from CDF. Linear when perturb == 0.
    if not perturb:
        u = torch.linspace(0., 1., n_samples, device=cdf.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples]) # [n_rays, n_samples]
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device=cdf.device) # [n_rays, n_samples]

    # Find indices along CDF where values in u would be placed.
    u = u.contiguous() # Returns contiguous tensor with same values.
    inds = torch.searchsorted(cdf, u, right=True) # [n_rays, n_samples]

    # Clamp indices that are out of bounds.
    below = torch.clamp(inds - 1, min=0)
    above = torch.clmap(inds, max=cdf.shape[-1] - 1)
    inds_g = torch.stack([below, above], dim=-1) # [n_rays, n_samples, 2]

    # Sample from cdf and the corresponding bin centers.
    matched_shape = list(inds_g.shape[:-1]) + [cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), dim=-1, index=inds_g)
    bins_g = torch.gather(bins.unsqueeze(-2).expand(matched_shape), dim=-1, index=inds_g)

    # Convert samples to ray length.
    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples # [n_rays, n_samples]

# Apply hierarchical sampling to the rays.
def sample_hierarchical(rays_o: torch.Tensor,
                        rays_d: torch.Tensor,
                        z_vals: torch.Tensor, 
                        weights: torch.Tensor,
                        n_samples: int,
                        perturb: bool = False
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    # Draw samples from PDF using z_vals as bins and weights as probabilities.
    z_vals_mid = .5 * (z_vals[..., 1] + z_vals[..., :-1])
    new_z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], n_samples, perturb=perturb)
    new_z_samples = new_z_samples.detach()

    # Resample points from ray based on PDF.
    z_vals_combined, _ =torch.sort(torch.cat([z_vals, new_z_samples], dim=-1), dim=-1)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_combined[..., :, None] # [n_rays, n_samples * 2, 3]

    return pts, z_vals_combined, new_z_samples

# ------------------------------ RAY CASTING ----------------------------------
# Find origin and direction of rays through every pixel and camera origin.
def get_rays(height: int, width: int, focal_length: float, c2w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

    # Apply pinhole camera model to gather directions at each pixel.
    i, j = torch.meshgrid(
        torch.arange(width, dtype=torch.float32).to(c2w),
        torch.arange(height, dtype=torch.float32).to(c2w),
        indexing='ij'
    )

    i, j = i.transpose(-1, -2), j.transpose(-1, -2)

    directions = torch.stack([(i - width * .5) / focal_length,
                              -(j - height * .5) / focal_length,
                              -torch.ones_like(i)
                              ], dim=-1)
    
    # Apply camera pose to directions.
    rays_d = torch.sum(directions[..., None, :] * c2w[:3, :3], dim=-1)

    # Origin is same for all directions (the optical center).
    rays_o = c2w[:3, -1].expand(rays_d.shape)

    return rays_o, rays_d

# Sample along ray from regularly-spaced bins.
def sample_stratified(rays_o: torch.Tensor, 
                      rays_d: torch.Tensor, 
                      near: float, 
                      far: float, 
                      n_samples: int, 
                      perturb: Optional[bool] = True, 
                      inverse_depth: bool = False
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    
    # Grab samples for space integration along ray.
    t_vals = torch.linspace(0., 1., n_samples, device=rays_o.device)

    # Sample linearly between 'near' and 'far'
    if not inverse_depth:
        z_vals = near * (1. - t_vals) + far * (t_vals)
    
    # Sample linearly in inverse depth (disparity)
    else:
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

    # Draw uniform samples from bins along ray.
    if perturb:
        mids = .5 * (z_vals[1:] + z_vals[:-1])
        upper = torch.concat([mids, z_vals[-1:]], dim=-1)
        lower = torch.concat([z_vals[:1], mids], dim=-1)
        t_rand = torch.rand([n_samples], device=z_vals.device)
        z_vals = lower + (upper - lower) * t_rand
    
    z_vals = z_vals.expand(list(rays_o.shape[:-1]) + [n_samples])

    # Apply scale from 'rays_d' and offset from 'rays_o' to samples
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    return pts, z_vals

# -------------------------------- HELPERS ------------------------------------
# Divide an input into chunks.
def get_chunks(inputs: torch.Tensor, chunksize: int = 2**15) -> List[torch.Tensor]:
    return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]

# Encode and chunkify points to prepare for NeRF model.
def prepare_chunks(points: torch.Tensor, encoding_function: Callable[[torch.Tensor], torch.Tensor], chunksize: int = 2**15) -> List[torch.Tensor]:

    points = points.reshape((-1, 3))
    points = encoding_function(points)
    points = get_chunks(points, chunksize=chunksize)

    return points

# Encode and chunkify view directions to prepare for NeRF model.
def prepare_viewdirs_chunks(points: torch.Tensor, 
                            rays_d: torch.Tensor, 
                            encoding_function: Callable[[torch.Tensor], torch.Tensor], 
                            chunksize: int=2**15
                           ) -> List[torch.Tensor]:
    
    # Prepare the view directions
    viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    viewdirs = viewdirs[:, None, ...].expand(points.shape).reshape((-1, 3))
    viewdirs = encoding_function(viewdirs)
    viewdirs = get_chunks(viewdirs, chunksize=chunksize)

    return viewdirs

# ----------------------------------- MAIN ------------------------------------
# Compute forward pass through model(s).
def nerf_forward(rays_o: torch.Tensor,
                 rays_d: torch.Tensor, 
                 near: float,
                 far: float,
                 encoding_fn: Callable[[torch.Tensor], torch.Tensor],
                 coarse_model: nn.Module,
                 kwargs_sample_stratified: dict = None,
                 n_samples_hierarchical: int = 0,
                 kwargs_sample_hierarchical: dict = None,
                 fine_model = None,
                 viewdirs_encoding_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 chunksize=2**15,
                 show=False
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:

    # Set no kwargs if none are given.
    if kwargs_sample_stratified is None:
        kwargs_sample_stratified = {}
    if kwargs_sample_hierarchical is None:
        kwargs_sample_hierarchical = {}

    # Sample query points along each ray.
    query_points, z_vals = sample_stratified(rays_o, rays_d, near, far, **kwargs_sample_stratified)

    # Prepare batches.
    batches = prepare_chunks(query_points, encoding_fn, chunksize=chunksize)
    if viewdirs_encoding_fn is not None:
        batches_viewdirs = prepare_viewdirs_chunks(query_points, rays_d, viewdirs_encoding_fn, chunksize=chunksize)
    else:
        batches_viewdirs = [None] * len(batches)

    # Coarse model pass.
    # Split the encoded points into 'chunks', run the model on all chunks, and concatenate the results (to avoid out-of-memory issues).
    predictions = []
    for batch, batch_viewdirs in zip(batches, batches_viewdirs):
        predictions.append(coarse_model(batch, viewdirs=batch_viewdirs, show=show))
    raw = torch.cat(predictions, dim=0)
    raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])

    print("Query Points: " + str(query_points[0][0]))

    temp = query_points.reshape(3, -1).detach().numpy()
    fig = plt.figure(figsize=(100,100))
    ax = plt.axes(projection="3d")
    ax.scatter3D(temp[0], temp[1], temp[2], color='green')
    plt.show()


    # Perform differentiable volume rendering to re-synthesize the RGB image.
    rgb_map, depth_map, acc_map, weights = raw2outputs(raw, z_vals, rays_d)

    # rgb_map, depth_map, acc_map, weights = render_volume_density(raw, rays_o, z_vals)
    outputs = { 'z_vals_stratified': z_vals }

    # Fine model pass.
    # Save previous outputs to return.
    if n_samples_hierarchical > 0:
        rgb_map_0, depth_map_0, acc_map_0 = rgb_map, depth_map, acc_map

        # Apply hierarchical sampling for fine query points.
        query_points, z_vals_combined, z_hierarch = sample_hierarchical(rays_o, rays_d, z_vals, weights, n_samples_hierarchical, **kwargs_sample_hierarchical)

        # Prepare inputs as before.
        batches = prepare_chunks(query_points, encoding_fn, chunksize=chunksize)
        if viewdirs_encoding_fn is not None:
            batches_viewdirs = prepare_viewdirs_chunks(query_points, rays_d, viewdirs_encoding_fn, chunksize=chunksize)
        else:
            batches_viewdirs = [None] * len(batches)    

        # Forward pass new samples through fine model.
        fine_model = fine_model if fine_model is not None else coarse_model
        predictions = []
        for batch, batch_viewdirs in zip(batches, batches_viewdirs):
            predictions.append(fine_model(batch, viewdirs=batch_viewdirs, show=show))

        raw = torch.cat(predictions)
        raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])

        # Perform differentiable volume rendering to re-synthesize the RGB image.
        rgb_map, depth_map, acc_map, weights = raw2outputs(raw, z_vals_combined, rays_d)

        # Store outputs.
        outputs['z_vals_hierarchical'] = z_hierarch
        outputs['rg_map_0'] = rgb_map_0
        outputs['depth_map_0'] = depth_map_0
        outputs['acc_map_0'] = acc_map_0

    # Store outputs.
    outputs['rgb_map'] = rgb_map
    outputs['depth_map'] = depth_map
    outputs['acc_map'] = acc_map
    outputs['weights'] = weights
    
    return outputs

# ------------------------------ TRAINING -------------------------------------
# Plot stratified and (optional) hierarchical samples.
def plot_samples(z_vals:torch.Tensor, z_hierarch: Optional[torch.Tensor] = None, ax: Optional[np.ndarray] = None):
    
    y_vals = 1 + np.zeros_like(z_vals)

    if ax is None:
        ax = plt.subplot()
    ax.plot(z_vals, y_vals, 'b-o')
    
    if z_hierarch is not None:
        y_hierarch = np.zeros_like(z_hierarch)
        ax.plot(z_hierarch, y_hierarch, 'r-o')

    ax.set_ylim([-1, -2])
    ax.set_title('Stratified Samples (blue) and Hierarchical Samples (red)')
    ax.axes.yaxis.set_visible(False)
    ax.grid(True)

    return ax

# Crop center square from image.
def crop_center(img: torch.Tensor, frac: float = 0.5) -> torch.Tensor:

    h_offset = round(img.shape[0] * (frac / 2))
    w_offset = round(img.shape[1] * (frac / 2))

    return img[h_offset:-h_offset, w_offset:-w_offset]

# Early stopping helper based on fitness criterion.
class EarlyStopping:
    def __init__(self, patience: int = 30, margin: float = 1e-4):
        self.best_fitness = 0.0
        self.best_iter = 0
        self.margin = margin
        self.patience = patience or float('inf')

    # Check if criterion for stopping is met.
    def __call__(self, iter: int, fitness: float):
        if (fitness - self.best_fitness) > self.margin:
            self.best_iter = iter
            self.best_fitness = fitness

        delta = iter - self.best_iter
        stop = delta >= self.patience # Stop training if patience exceeded.

        return stop
    
# Initialize models, encoders, and optimizer for NeRF training.
def init_models():

    # Encoders.
    encoder = PositionalEncoder(d_input, n_freqs, log_space=log_space)
    encode = lambda x: encoder(x)

    # View direction encoders.
    if use_viewdirs:
        encoder_viewdirs = PositionalEncoder(d_input, n_freqs_views, log_space=log_space)
        encode_viewdirs = lambda x: encoder_viewdirs(x)
        d_viewdirs = encoder_viewdirs.d_output
    else:
        encode_viewdirs = None
        d_viewdirs = None

    # Models.
    model = NeRF(encoder.d_output, n_layers=n_layers, d_filter=d_filter, skip=skip, d_viewdirs=d_viewdirs)
    model.to(device)
    model_params = list(model.parameters())

    if use_fine_model:
        fine_model = NeRF(encoder.d_output, n_layers=n_layers, d_filter=d_filter, skip=skip, d_viewdirs=d_viewdirs)
        fine_model.to(device)
        model_params = model_params + list(fine_model.parameters())
    else:
        fine_model = None

    # Optimizer.
    optimizer = Adam(model_params, lr=lr)

    # Early stopping.
    warmup_stopper = EarlyStopping(patience=50)

    return model, fine_model, encode, encode_viewdirs, optimizer, warmup_stopper

# Launch training session for NeRF.
def train(model: NeRF, encode: PositionalEncoder, fine_model: NeRF, encode_viewdirs: PositionalEncoder, optimizer: Adam, warmup_stopper: EarlyStopping):

    # Shuffle rays across all images.
    if not one_image_per_step:
        height, width = images.shape[1:3]
        all_rays = torch.stack([torch(get_rays(height, width, focal, p), 0) for p in poses[:n_training]], 0)
        rays_rgb = torch.cat([all_rays, images[:, None]], 1)
        rays_rgb = torch.permute(rays_rgb, [0, 2, 3, 1, 4])
        rays_rgb = rays_rgb.reshape([-1, 3, 3])
        rays_rgb = rays_rgb.type(torch.float32)
        rays_rgb = rays_rgb[torch.randperm(rays_rgb.shape[0])]
        i_batch = 0

    train_psnrs = []
    val_psnrs = []
    iternums = []
    for i in trange(n_iters):
        model.train()

        # Randomly pick an image as the target.
        if one_image_per_step:
            target_img_idx = np.random.randint(images.shape[0])
            target_img = images[target_img_idx].to(device)

            if center_crop and i < center_crop_iters:
                target_img = crop_center(target_img)

            height, width = target_img.shape[:2]
            target_pose = poses[target_img_idx].to(device)
            rays_o, rays_d = get_rays(height, width, focal, target_pose)
            rays_o = rays_o.reshape([-1, 3])
            rays_d = rays_d.reshape([-1, 3])
        # Random over all images.
        else:
            batch = rays_rgb[i_batch:i_batch + batch_size]
            batch = torch.transpose(batch, 0, 1)

            rays_o, rays_d, target_img = batch
            height, width = target_img.shape[:2]
            i_batch += batch_size

            # Shuffle after one epoch.
            if i_batch >= rays_rgb.shape[0]:
                rays_rgb = rays_rgb[torch.randperm(rays_rgb.shape[0])]
                i_batch = 0
        
        target_img = target_img.reshape([-1, 3])

        # Run one iteration of TinyNeRF and get the rendered RGB image.
        outputs = nerf_forward(rays_o, rays_d, near, far, encode, model, 
                               kwargs_sample_stratified=kwargs_sample_stratified, 
                               kwargs_sample_hierarchical=kwargs_sample_hierarchical, 
                               fine_model=fine_model, 
                               viewdirs_encoding_fn=encode_viewdirs, 
                               chunksize=chunksize)

        # Check for any numerical issues.
        for k, v in outputs.items():
            if torch.isnan(v).any():
                print(f"! [Numerical Alert] {k} contains NaN.")
            if torch.isinf(v).any():
                print(f"! [Numerical Alert] {k} contains Inf.")

        # Backprop.
        rgb_predicted = outputs['rgb_map']
        loss = F.mse_loss(rgb_predicted, target_img)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        psnr = -10. * torch.log10(loss)
        train_psnrs.append(psnr.item())

        # Evaluate test_img at given display rate.
        if i % display_rate == 0 and i != 0:
            model.eval()
            height, width = test_img.shape[:2]
            rays_o, rays_d = get_rays(height, width, focal, test_pose)
            rays_o = rays_o.reshape([-1, 3])
            rays_d = rays_d.reshape([-1, 3])
            outputs = nerf_forward(rays_o, rays_d, near, far, encode, model, 
                                   kwargs_sample_stratified=kwargs_sample_stratified,
                                   kwargs_sample_hierarchical=kwargs_sample_hierarchical,
                                   fine_model=fine_model,
                                   viewdirs_encoding_fn=encode_viewdirs,
                                   chunksize=chunksize, 
                                   show=True)
            
            rgb_predicted = outputs['rgb_map']
            loss = F.mse_loss(rgb_predicted, test_img.reshape(-1, 3))
            print("Loss: ", loss.item())
            val_psnr = -10. * torch.log10(loss)
            val_psnrs.append(val_psnr.item())
            iternums.append(i)

            # Plot example outputs.
            fig, ax = plt.subplots(1, 4, figsize=(24, 4), gridspec_kw={'width_ratios': [1, 1, 1, 3]})
            ax[0].imshow(rgb_predicted.reshape([height, width, 3]).detach().cpu().numpy())
            ax[0].set_title(f'Iteration: {i}')
            ax[1].imshow(test_img.detach().cpu().numpy())
            ax[1].set_title(f'Target')
            ax[2].plot(range(0, i + 1), train_psnrs, 'r')
            ax[2].plot(iternums, val_psnrs, 'b')
            ax[2].set_title('PSNR (train=red, val=blue)')
            z_vals_strat = outputs['z_vals_stratified'].view((-1, n_samples))
            z_sample_strat = z_vals_strat[z_vals_strat.shape[0] // 2].detach().cpu().numpy()

            if 'z_vals_hierarchical' in outputs:
                z_vals_hierarch = outputs['z_vals_hierarchical'].view((-1, n_samples_hierarchical))
                z_sample_hierarch = z_vals_hierarch[z_vals_hierarch.shape[0] // 2].detach().cpu().numpy()
            else:
                z_sample_hierarch = None
            
            _ = plot_samples(z_sample_strat, z_sample_hierarch, ax=ax[3])
            ax[3].margins(0)
            plt.show()

        # Check PSNR for issues and stop if any are found.
        if i == warmup_iters - 1:
            if val_psnr < warmup_min_fitness:
                print(f'Val PSNR {val_psnr} below warmup_min_fitness {warmup_min_fitness}. Stopping...')
                return False, train_psnrs, val_psnrs
        elif i < warmup_iters:
            if warmup_stopper is not None and warmup_stopper(i, psnr):
                print(f'Train PSNR flatlined at {psnr} for {warmup_stopper.patience} iters. Stopping...')
                return False, train_psnrs, val_psnrs
            
    return True, train_psnrs, val_psnrs

data = np.load('tiny_nerf_data.npz')
images = data['images']
poses = data['poses']
focal = data['focal']

print(images.shape)
print(poses.shape)
print(focal.shape)

height, width = images.shape[1:3]
near, far = 2., 6.

n_training = 100
test_img_idx = 101
test_img, test_pose = images[test_img_idx], poses[test_img_idx]

print(type(test_img))

plt.imshow(test_img)
plt.show()

print('Pose')
print(test_pose)

dirs = np.stack([np.sum([0, 0, -1] * pose[:3, :3], axis=-1) for pose in poses])
origins = poses[:, :3, -1]

ax = plt.figure(figsize=(12, 8)).add_subplot(projection='3d')
_ = ax.quiver(
    origins[..., 0].flatten(),
    origins[..., 1].flatten(),
    origins[..., 2].flatten(),
    dirs[..., 0].flatten(),
    dirs[..., 1].flatten(),
    dirs[..., 2].flatten(),
    length=0.5,
    normalize=True
)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Gather as torch tensors.
images = torch.from_numpy(data['images'][:n_training]).to(device)
poses = torch.from_numpy(data['poses']).to(device)
focal = torch.from_numpy(data['focal']).to(device)
test_img = torch.from_numpy(data['images'][test_img_idx]).to(device)
test_pose = torch.from_numpy(data['poses'][test_img_idx]).to(device)

# Grab rays from sample image.
height, width = images.shape[1:3]
with torch.no_grad():
    ray_origin, ray_direction = get_rays(height, width, focal, test_pose)

print('Ray Origin')
print(ray_origin.shape)
print(ray_origin[height // 2, width // 2, :])
print('')

print('Ray Direction')
print(ray_direction.shape)
print(ray_direction[height // 2, width // 2, :])
print('')

# Draw stratified samples from example.
rays_o = ray_origin.view(-1, 3)
rays_d = ray_direction.view([-1, 3])
n_samples = 8
perturb = True
inverse_depth = False
with torch.no_grad():
    pts, z_vals = sample_stratified(rays_o, rays_d, near, far, n_samples, perturb=perturb, inverse_depth=inverse_depth)

print('Input Points')
print(pts.shape)
print('')
print('Distances Along Ray')
print(z_vals.shape)

y_vals = torch.zeros_like(z_vals)

_, z_vals_unperturbed = sample_stratified(rays_o, rays_d, near, far, n_samples, perturb=False, inverse_depth=inverse_depth)

plt.plot(z_vals_unperturbed[0].cpu().numpy(), 1 + y_vals[0].cpu().numpy(), 'b-o')
plt.plot(z_vals[0].cpu().numpy(), y_vals[0].cpu().numpy(), 'r-o')
plt.ylim([-1, 2])
plt.title('Stratified Sampling (blue) with Perturbation (red)')
ax = plt.gca()
ax.axes.yaxis.set_visible(False)
plt.grid(True)
plt.show()

# Create encoders for points and view directions.
encoder = PositionalEncoder(d_input=3, n_freqs=10)
viewdirs_encoder = PositionalEncoder(d_input=3, n_freqs=4)

# Grab flattened points and view directions.
pts_flattened = pts.reshape(-1, 3)
viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
flattened_viewdirs = viewdirs[:, None, ...].expand(pts.shape).reshape((-1, 3))

# Positional encode inputs.
encoded_points = encoder(pts_flattened)
encoded_viewdirs = viewdirs_encoder(flattened_viewdirs)

print('Encoded Points')
print(encoded_points.shape)
print(torch.min(encoded_points), torch.max(encoded_points), torch.mean(encoded_points))
print()

print('Encoded View Directions')
print(encoded_viewdirs.shape)
print(torch.min(encoded_viewdirs), torch.max(encoded_viewdirs), torch.mean(encoded_viewdirs))
print()

# Hyperparameters.
# Encoders.
d_input = 3
n_freqs = 10
log_space = True
use_viewdirs = True
n_freqs_views = 4

# Stratified sampling.
n_samples = 64
perturb = True
inverse_depth = False

# Model.
d_filter = 128
n_layers = 2
skip = []
use_fine_model = True
d_filter_fine = 128
n_layers_fine = 6

# Hierarchical sampling.
n_samples_hierarchical = 64
perturb_hierarchical = False

# Optimizer.
lr = 5e-4

# Training
n_iters = 1000
batch_size = 2**14
one_image_per_step = True
chunksize = 2**14
center_crop = True
center_crop_iters = 50
display_rate = 25

# Early stopping.
warmup_iters = 100
warmup_min_fitness = 10.0
n_restarts = 10

# We bundle the kwargs for various functions to pass all at once.
kwargs_sample_stratified = {
    'n_samples': n_samples,
    'perturb': perturb,
    'inverse_depth': inverse_depth
}
kwargs_sample_hierarchical = {
    'perturb': perturb
}

# Run training session(s).
for _ in range(n_restarts):
    model, fine_model, encode, encode_viewdirs, optimizer, warmup_stopper = init_models()
    success, train_psnrs, val_psnrs = train(model, encode, fine_model, encode_viewdirs, optimizer, warmup_stopper)
    if success and val_psnrs[-1] >= warmup_min_fitness:
        print('Training successful.')
        break

print()
print(f'Done.')