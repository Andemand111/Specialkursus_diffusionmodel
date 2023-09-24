#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
#%%


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cpu"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x
#%%

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        
    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)



class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cpu"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output
#%%
# generate a model on CPU and a diffusion object and try to run a forward pass

class Model(nn.Module):
    def __init__(self, dimensions, time_steps, beta_start, beta_end):
        super().__init__()

        self.dimensions = dimensions
        self.img_size = torch.tensor(dimensions).prod()
        self.time_dim = dimensions[1] * dimensions[2]

        self.downsample = nn.Sequential(
            nn.Unflatten(1, (4, 64, 64)),

            DoubleConv(64, 128),
            Down(48, 64),
            SelfAttention(64, 16),
            DoubleConv(48, 48*4),
            Down(32, 48),
            SelfAttention(48, 12),
            DoubleConv(32, 32*8),
            Down(16, 32),
            SelfAttention(32, 8),

            nn.Flatten()
        )

        self.upsample = nn.Sequential(
            nn.Unflatten(1, (256, 4, 4)),

            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2),

            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2),

            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2),

            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2),

            nn.ConvTranspose2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.Flatten()
        )

        self.time_steps = time_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = torch.linspace(beta_start, beta_end, time_steps)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha = torch.sqrt(self.alpha)
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        self.one_minus_alpha_hat = 1 - self.alpha_hat
        self.sqrt_one_minus_alpha_hat = torch.sqrt(self.one_minus_alpha_hat)

        self.sigma_sq = (
            1 - self.alpha) * torch.roll(self.one_minus_alpha_hat, 1) / self.one_minus_alpha_hat
        self.sigma_sq[0] = 0
        self.sigma = torch.sqrt(self.sigma_sq)

    def time_encoding(self, t):
        frequencies = torch.arange(0, self.time_dim, 2) * torch.pi / self.time_steps
        angles = t.unsqueeze(1) * frequencies.unsqueeze(0)
        encodings = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

        return encodings

    def forward(self, x, t):
        y = torch.cat((x, t), 1)
        y = self.downsample(y)
        y = self.upsample(y)

        return y
    

#%%
    # make dummy datapoint and run forward pass through Model
model = Model((3, 64, 64), 1000, 0.0001, 0.02)
x = torch.randn((1, 3, 64, 64))
sample = model(x, t)
print(sample.shape)

#%%


#%%

def train(self, epochs, dataloader, filename="diffusion_model"):
    optimizer = torch.optim.Adam(self.parameters())
    cost = nn.MSELoss()
    batch_size = dataloader.batch_size

    for epoch in range(epochs):
        losses = torch.zeros(len(dataloader))

        for i, x_0 in enumerate(tqdm(dataloader)):
            ts = self.sample_time_steps(batch_size)
            x_t, eps = self.make_noisy_image(x_0, ts)
            time_encodings = self.time_encoding(ts)
            pred = self(x_t, time_encodings)
            loss = cost(pred, x_0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses[i] = loss.item()

        print(f"Epoch: {epoch}")
        print(losses.mean().item())
        self.save_model(filename)

        self.sample_and_show_image(f"Result after epoch: {epoch}")
        self.view_reconstructed_images(x_0[0], 200)
        self.view_reconstructed_images(x_0[0], 500)
        self.view_reconstructed_images(x_0[0], 800)

def make_noisy_image(self, x, t):
    t = t.view(-1, 1)
    eps = torch.randn(x.shape)
    x_t = self.sqrt_alpha_hat[t] * x + \
        self.sqrt_one_minus_alpha_hat[t] * eps

    return x_t, eps

def sample_time_steps(self, n):
    return torch.randint(0, self.time_steps - 1, (n,))

def save_model(self, filename):
    torch.save(self.state_dict(), filename)
    print("Model saved!")

def load_model(self, filename):
    self.load_state_dict(torch.load(filename, map_location="cpu"))
    print("Model loaded!")

def reconstruct_noisy_image(self, image, t):
    t = torch.tensor(t).view((1,))
    enc = self.time_encoding(t)
    noisy, _ = self.make_noisy_image(image.view((1, -1)), t)
    reconstructed = self(noisy.view(1, -1), enc)
    return noisy.detach(), reconstructed.detach()

def view_reconstructed_images(self, x, t):
    noisy, reconstructed = self.reconstruct_noisy_image(x, t)
    noisy = torch.clamp(noisy, 0, 1)
    fig, axs = plt.subplots(1, 3)
    fig.suptitle(f'Timestep {t}', fontsize=16)
    axs[0].imshow(x.view(*self.dimensions).permute(1, 2, 0))
    axs[0].set_title("Original")
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].imshow(noisy.view(*self.dimensions).permute(1, 2, 0))
    axs[1].set_title("Noisy")
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[2].imshow(reconstructed.view(*self.dimensions).permute(1, 2, 0))
    axs[2].set_title("Reconstructed")
    axs[2].set_xticks([])
    axs[2].set_yticks([])

    plt.show()

def sample_image(self, x_T = None):
    x_t = torch.randn((1, self.img_size)) if x_t == None else x_T
    for t in reversed(range(1, self.time_steps)):
        t_enc = self.time_encoding(torch.tensor(t).view((1,)))
        x_0 = self(x_t, t_enc)
        k1 = self.sqrt_alpha[t] * self.one_minus_alpha_hat[t - 1] * x_t
        k2 = self.sqrt_alpha_hat[t - 1] * (1 - self.alpha[t]) * x_0
        k3 = self.one_minus_alpha_hat[t]
        k4 = self.sigma[t] * torch.randn_like(self.sigma[t])
        x_t = (k1 + k2) / k3
        if t > 1:
            x_t += k4

    x_t = torch.clamp(x_t, 0, 1)
    return x_t.detach()

def sample_and_show_image(self, title=""):
    x_t = self.sample_image()
    plt.imshow(x_t.view(*self.dimensions).permute(1, 2, 0), cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()

model = Model((3, 64, 64), 1000, 0.0001, 0.02)

