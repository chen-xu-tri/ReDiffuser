import numpy as np
import matplotlib.pyplot as plt
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import math
import random



# Number of samples per Gaussian
num_samples = 5000

# Mean positions of the Gaussians forming a square
means = np.array([
    [1, 1],
    [-1, 1],
    [-1, -1],
    [1, -1]
])

# Covariance matrix (same for all Gaussians for simplicity)
covariance = np.eye(2) * 0.02

# Generate data points from each Gaussian
data = []
for mean in means:
    samples = np.random.multivariate_normal(mean, covariance, num_samples)
    data.append(samples)

# Concatenate data from all Gaussians into a single array
data = np.concatenate(data, axis=0)

# Convert data to torch tensors
data_tensor = torch.tensor(data, dtype=torch.float32)

# Define a DataLoader
batch_size = 512
data_loader = DataLoader(data_tensor, batch_size=batch_size, shuffle=True)

class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class DiffusionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DiffusionMLP, self).__init__()
        dsed = 16
        self.fc1 = nn.Linear(input_dim+dsed, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)


        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
            )

    def forward(self, x,t):
        t = t.expand(x.shape[0])
        t = self.diffusion_step_encoder(t)
        x = torch.cat([x,t],dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Parameters
input_dim = 2  # Dimension of input data (X, Y)
hidden_dim = 128  # Hidden layer dimension
output_dim = 2  # Dimension of output (X, Y)

num_components = 5

# Create the model
net = DiffusionMLP(input_dim, hidden_dim, output_dim)
rnd_target = MLP(output_dim, hidden_dim, 32)
rnd_net = MLP(output_dim, hidden_dim, 32)

num_diffusion_iters = 10


noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule='squaredcos_cap_v2',
        # clip output to [-1,1] to improve stability
        clip_sample=False,
        # our network predicts noise (instead of denoised action)
        prediction_type='epsilon'
    )

noise_scheduler.set_timesteps(num_diffusion_iters)

optimizer = torch.optim.AdamW(
    params=net.parameters(),
    lr=1e-4, weight_decay=1e-6)


rnd_optimizer = torch.optim.AdamW(
    params=rnd_net.parameters(),
    lr=1e-4, weight_decay=1e-6)


num_epochs = 500
with tqdm(range(num_epochs), desc='Epoch') as tglobal:
    # epoch loop
    for epoch_idx in tglobal:
        epoch_loss = list()
        # batch loop
        with tqdm(data_loader, desc='Batch', leave=False) as tepoch:
            for batch in tepoch:

                noise = torch.randn(batch.shape)

                # sample a diffusion iteration for each data point
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (batch.shape[0],), 
                ).long()

                # add noise to the clean images according to the noise magnitude at each diffusion iteration
                # (this is the forward diffusion process)
                noisy_data = noise_scheduler.add_noise(
                    batch, noise, timesteps)
                label = noise

                optimizer.zero_grad()

                # noisy_data = ensemble[0](noisy_data)
                output = net(noisy_data, timesteps)

                # L2 loss
                loss = nn.functional.mse_loss(output, label)
                loss.backward()
                optimizer.step()

                # Train RND
                target = rnd_target(batch)
                pred   = rnd_net(batch)
                rnd_loss = nn.functional.mse_loss(pred, target)
                rnd_loss.backward()
                rnd_optimizer.step()


num_eval_samples = 5000
noise = torch.randn((num_eval_samples, 2))

sample = noise

sample_history = []
for k in noise_scheduler.timesteps:
    # noise_pred = net(ensemble[0](sample), k)
    noise_pred = net(sample, k)
    sample = noise_scheduler.step(
        model_output=noise_pred,
        timestep = k,
        sample=sample
    ).prev_sample
    sample_history.append(sample.detach().numpy())




pred = rnd_net(sample)
true = rnd_target(sample)
diff = torch.mean(torch.abs(true-pred), dim=1)

# sample = sample.detach().numpy()
min_vals, _ = torch.min(diff, dim=0)
max_vals, _ = torch.max(diff, dim=0)
normalized_scores = (diff - min_vals) / (max_vals - min_vals)
sample = sample.detach().numpy()
normalized_scores = normalized_scores.detach().numpy()


plt.figure(figsize=(8, 8))
# plt.scatter(noise[:, 0], noise[:, 1], s=10, alpha=0.1, c="r")
# plt.scatter(data[:, 0], data[:, 1], s=10, alpha=0.5)
plt.title('Sampled Data from Four Equally Spaced Gaussians')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.axis('equal')
# plt.scatter(sample[:, 0], sample[:, 1], s=10, alpha=0.5, c="g")
plt.scatter(sample[:, 0], sample[:, 1], s=10, alpha=0.5, c=normalized_scores, cmap='viridis')
plt.show()
plt.show()