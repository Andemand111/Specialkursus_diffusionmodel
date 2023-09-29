# %%
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset
import torch

from ResNETWithAttention_utilstester import MNIST, CIFAR10

train_set = MNIST(n = 10_000)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dimensions = train_set[0].shape
img_size = torch.prod(torch.tensor(dimensions))

# %%
## Visualize the data
import matplotlib.pyplot as plt
import numpy as np

# show multiple images in a 
# grid format
fig, axs = plt.subplots(3, 3, figsize=(8,8))
for i in range(9):
    img = train_set[i]
    img = img * 0.5 + 0.5
    axs[i//3, i%3].imshow(img.view(*dimensions).permute(1,2,0), cmap='gray')
    axs[i//3, i%3].axis('off')


# %%
## make noisy images

from ResNETWithAttention_utilstester import NoiseSchedule

beta_start = 1e-4
beta_end = 0.02
T = 500
noise_schedule = NoiseSchedule(beta_start, beta_end, T)

fig, axs = plt.subplots(1, 11, figsize=(20, 2))
x0 = train_set[0].flatten()
ts = torch.linspace(0, T - 1, 11).view(-1, 1).long()
xts, _, _ = noise_schedule.make_noisy_images(x0, ts)
for i, t in enumerate(ts):
    img = xts[i].view(*dimensions).permute(1, 2, 0).numpy()
    img = (img + 1) / 2
    img = np.clip(img, 0, 1)
    axs[i].imshow(img, cmap='gray')
    axs[i].axis('off')
    axs[i].set_title(f't={t.item() + 1}')
plt.savefig("figures/noisy_images.png", dpi=300, bbox_inches='tight')
plt.show()


# %% [markdown]
# A better way of understanding the noise scheduling is to examine what happens to the value of a single pixel.
# As seen below, the value of the pixel is approching 0 and the standard deviation is approaching 1. In other words, the pixel is becoming closer to a standard normal distribution. This is the goal of the noise scheduling. 
# However, the distribution only approaches the standard normal distribution, it does not reach it. This is because the noise is not added to the pixel, but rather multiplied by the pixel. This means that the pixel will never reach 0, but rather approach it. The values of $\mu$ and $\sigma$ for each pixel distribution are shown below.

# %%
def normal_dist(x, mu, std):
    return 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2 * std**2))
pixel_value = 0.5
fig, axs = plt.subplots(1, 11, figsize=(20, 2))
mus = noise_schedule.sqrt_alpha_hat[ts]
stds = noise_schedule.sqrt_one_minus_alpha_hat[ts]
xx = torch.linspace(-2, 2, 1000)
for i, t in enumerate(ts):
    mu = mus[i] * pixel_value
    std = stds[i]
    axs[i].plot(xx, normal_dist(xx, mu, std))
    axs[i].set_title(f't={t.item() + 1}')
    axs[i].set_ylim(0, 6)
    axs[i].set_xlim(-1, 1)
    axs[i].axvline(0, color='b', linestyle='--')
    if i % 2 == 0:
        axs[i].set_xlabel("$\mu$ = {:.2f}, $\sigma$ = {:.2f}".format(round(mu.item(), 2), round(std.item(), 2)))
    if i == 0:
        axs[i].set_ylabel('$q(x_t | x_0)$')

plt.savefig("figures/pixel_distributions.png", dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# To show that the distribution of the final layer actually approaches a standard normal distribution, we've plotted the KL-divergence between each layer and a standard normal. The KL-divergence clearly approaches 0. 

# %%
from tqdm import tqdm

## this small image is just a 10x10 patch of the original image

KL_list = np.zeros(T)
for t in range(T):
    mu = noise_schedule.sqrt_alpha_hat[t] * x0.flatten()
    std = noise_schedule.sqrt_one_minus_alpha_hat[t]

    k1 = img_size * std ** 2
    k2 = torch.sum(mu ** 2)
    k3 = - 2 * img_size * torch.log(std)
    kl = 1/2 * (k1 + k2 + k3 - img_size)
    
    KL_list[t] = kl.detach().numpy()

plt.plot(KL_list)
txt = "KL divergence between noisy distribution and prior, \n with " + r"$\beta$" + " range [0.0001, 0.02] and T = {}"
plt.title(txt.format(T))
plt.xlabel("t")
plt.ylabel("KL divergence")
plt.hlines(0, 0, T, color='r', linestyle='--')
plt.show()


# %% [markdown]
# We somehow have to encode the different timesteps. Here, we'll use sinusoidal encoding. 
# 

# %%
from ResNETWithAttention_utilstester import SinusoidalEmbeddings

ts = np.arange(0, T)
embeddings = SinusoidalEmbeddings(torch.tensor(ts).float())
plt.imshow(embeddings, cmap='viridis', aspect='auto')
plt.ylabel("timestep")
plt.xlabel("d")
plt.title("Sinusoidal embeddings")
plt.savefig("figures/sinusoidal_embeddings.png", dpi=300, bbox_inches='tight')
plt.show()


# %%
from ResNETWithAttention_utilstester import SimpleModel, ResNET
import torch.nn as nn
mse = nn.MSELoss()

network_args = [dimensions[0], 64, [64,128,256]]
simple_network = ResNET(*network_args).to(device)
args = [simple_network, train_loader, noise_schedule, dimensions, device]
simple_model = SimpleModel(*args)
simple_model.load_model()

# %%
def sample_and_show_image(model, title="Sampled image"):
    xt = model.sample()
    xt = (xt - xt.min()) / (xt.max() - xt.min())
    img = xt.view(*dimensions).permute(1, 2, 0).cpu().numpy()
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# %%
def training_loop(model, epochs):
    optimizer = torch.optim.Adam(model.parameters())
    epoch_loss = np.zeros(epochs)

    num_paramters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    parameters = np.zeros((epochs, num_paramters))

    for epoch in range(epochs):
        losses = torch.zeros(len(train_loader))

        for i, x0 in enumerate(tqdm(train_loader)):
            loss = model.loss(x0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses[i] = loss.item()


        print(f"Epoch: {epoch}")
        print(losses.mean().item())
        # sample_and_show_image(model, "Epoch {}".format(epoch))
        #model.save_model()
        torch.save(model.state_dict(), "C:/Users/gusta/OneDrive/Skrivebord/KI & Data/Semester 5/Diffusionsprojekt/simple_model.pt")
        epoch_loss[epoch] = losses.mean().item()

        curr_params = np.concatenate([p.detach().cpu().numpy().flatten() for p in model.parameters() if p.requires_grad])
        parameters[epoch] = curr_params
    
    return epoch_loss, parameters

# %%
simple_model_losses, simple_model_parameters = training_loop(simple_model, 1)
np.save("C:/Users/gusta/OneDrive/Skrivebord/KI & Data/Semester 5/Diffusionsprojekt/simple_model_losses.npy", simple_model_losses)
np.save("C:/Users/gusta/OneDrive/Skrivebord/KI & Data/Semester 5/Diffusionsprojekt/simple_model_parameters.npy", simple_model_parameters)

# %%
from ResNETWithAttention_utilstester import MuModel
mu_network = ResNET(*network_args).to(device)
args = [mu_network, train_loader, noise_schedule, dimensions, device]
mu_model = MuModel(*args)
mu_model.load_model()

mu_model_losses, mu_model_parameters = training_loop(mu_model, 20)
np.save("../data/mu_model_losses.npy", mu_model_losses)
np.save("../data/mu_model_parameters.npy", mu_model_parameters)

# %%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
pca = PCA(n_components=2)

simple_model_parameters = np.load("../data/simple_model_parameters.npy")
simple_model_parameters = scaler.fit_transform(simple_model_parameters)
pca.fit(simple_model_parameters)

# %%
def make_state_dict_from_paramters(model, parameters):
    state_dict = model.state_dict()
    start = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            end = start + param.numel()
            state_dict[name] = parameters[start:end].view(param.shape).to(device)
            start = end
    return state_dict

# %%
from mpl_toolkits import mplot3d

temp_network = ResNET(*network_args).to(device)
temp_model = SimpleModel(temp_network, train_loader, noise_schedule, dimensions, device)
resolution = 30
xx, yy = np.meshgrid(np.linspace(-3, 3, resolution), np.linspace(-3, 3, resolution))
zz = np.zeros_like(xx.flatten())

for i, (x, y) in tqdm(enumerate(zip(xx.flatten(), yy.flatten()))):
    params = np.array([x, y])
    params = pca.inverse_transform(params)
    params = scaler.inverse_transform(params.reshape(1, -1))
    params = torch.from_numpy(params).float().to(device)
    state_dict = make_state_dict_from_paramters(temp_model, params[0])
    temp_model.load_state_dict(state_dict)
    x0 = train_set[:64].view(64, *dimensions)
    loss = temp_model.loss(x0).item()
    zz[i] = loss

# %%
%matplotlib inline

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx, yy, zz.reshape(resolution, resolution), cmap='viridis')


