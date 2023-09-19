import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

class Model(nn.Module):
    def __init__(self, dimensions, time_steps, beta_start, beta_end):
        super().__init__()

        self.dimensions = dimensions
        self.img_size = torch.tensor(dimensions).prod()
        self.time_dim = dimensions[1] * dimensions[2]

        self.downsample = nn.Sequential(
            nn.Unflatten(1, (2, 28, 28)),
            nn.Conv2d(2, 16, kernel_size=5, stride=1, padding=2), 
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.2),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.LeakyReLU(),
        )

        self.upsample = nn.Sequential(
            nn.Linear(512, 7 * 7 * 64),
            nn.LeakyReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=5, stride=1, padding=2),  
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(8),
            nn.Dropout2d(0.2),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(8 * 28 * 28, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, self.img_size),
            nn.Tanh(),
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

        self.sigma_sq = (1 - self.alpha) * torch.roll(self.one_minus_alpha_hat, 1) / self.one_minus_alpha_hat
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
            print("loss: ", losses.mean().item())
            self.save_model(filename)

            self.sample_and_show_image(f"Result after epoch: {epoch}")

    def make_noisy_image(self, x_0, t):
        t = t.view(-1, 1)
        eps = torch.randn(x_0.shape)
        mu = self.sqrt_alpha_hat[t] * x_0
        std  = self.sqrt_one_minus_alpha_hat[t]
        x_t = mu + std * eps

        return x_t, eps

    def sample_time_steps(self, n):
        return torch.randint(1, self.time_steps - 1, (n,))

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)
        print("Model saved!")

    def load_model(self, filename):
        self.load_state_dict(torch.load(filename, map_location="cpu"))
        print("Model loaded!")

    def get_prior_sample(self, x_t, t):
        t_enc = self.time_encoding(torch.tensor(t).view((1,)))
        x_0 = self(x_t, t_enc)
        k1 = self.sqrt_alpha[t] * self.one_minus_alpha_hat[t - 1] * x_t
        k2 = self.sqrt_alpha_hat[t - 1] * (1 - self.alpha[t]) * x_0
        x_t = (k1 + k2) / self.one_minus_alpha_hat[t]
        if t > 1:
            x_t += torch.randn_like(x_t) * self.sigma[t]

        return x_t

    def sample_image(self, x_T = None):
        x_t = torch.randn((1, self.img_size)) if x_T == None else x_T
        for t in reversed(range(1, self.time_steps)):
            x_t = self.get_prior_sample(x_t, t)

        x_t = x_t * 0.5 + 0.5
        x_t = torch.clamp(x_t, 0, 1)
        return x_t.detach()

    def sample_and_show_image(self, title=""):
        x_t = self.sample_image()
        plt.imshow(x_t.view(*self.dimensions).permute(1, 2, 0), cmap="gray")
        plt.title(title)
        plt.axis("off")
        plt.show()