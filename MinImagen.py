import torch
import torch.nn as nn
import torch.nn.functional as F

def exists(val):
    return val is not None

# Returns the input value 'val' unless it is 'None', in which case the default 'd' is returned if it is a value or
# d() if it is a callable.
def default(val, d):
    
    if exists(val):
        return val
    return d() if callable(d) else d

# Extracts values from a using t as indices
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1, ) * (len(x_shape) - 1)))

class GaussianDiffusion(nn.Module):

    def __init__(self, *, timesteps):
        super().__init__()

        # Timesteps < 20 => scale > 50 => beta_end > 1 => alphas[=1] < 0 => sqrt_alphas_cumprod[-1] is NaN
        assert not timesteps < 20, f'timesteps must be at least 20'
        self.num_timesteps = timesteps

        # Create variance schedule - specifies the variance of the Gaussian noise at a given timestep.
        scale = 1000 / timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

        # Diffusion model constants/buffers.
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        # Register buffer helper function.
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32), persistent=False)

        # Register variance schedule related buffers.
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Buffer for diffusion calculations q(x_t | x_{t-1}) and others.
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)

        # Clipped because posterior variance is 0 at the beginning of the diffusion chain.
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))

        # Buffers for calculating the q_posterior mean.
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    # Sample from q at given timestep
    def q_sample(self, x_start, t, noise):

        noise = default(noise, lambda: torch.randn_like(x_start))

        noised = (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

        return noised
    
    # Given a noised image and its noise component, calculate the unnoised image
    def predict_start_from_noise(self, x_t, t, noise):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise)
    
    def q_posterior(self, x_start, x_t, t):
        

print(f'PyTorch version: {torch.__version__}')
print('*'*10)
print(f'_CUDA version: ')
#!nvcc --version
print('*'*10)
print(f'CUDNN version: {torch.backends.cudnn.version()}')
print(f'Available GPU devices: {torch.cuda.device_count()}')
print(f'Device Name: {torch.cuda.get_device_name()}')