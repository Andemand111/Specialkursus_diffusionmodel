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
            nn.Unflatten(1, (4, 64, 64)),

            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(), 
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten()
        )

        self.linear = nn.Sequential(
            nn.Linear(256 * 4 * 4, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 256 * 4 * 4),
            nn.ReLU()
        )
        
        self.upsample = nn.Sequential(
            nn.Unflatten(1, (256, 4, 4)),

            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
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
        
        self.sigma_sq = (1 - self.alpha) * torch.roll(self.one_minus_alpha_hat, 1) / self.one_minus_alpha_hat
        self.sigma_sq[0] = 0
        self.sigma = torch.sqrt(self.sigma_sq)
        
    def time_encoding(self, t):
        frequencies = torch.arange(0, self.time_dim, 2) * torch.pi / self.time_steps
        angles = t.unsqueeze(1) * frequencies.unsqueeze(0)
        encodings = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        
        return encodings     
    
    def forward(self, x, t):
        y = torch.cat((x,t), 1)
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
            print(losses.mean().item())
            self.save_model(filename)
        
            self.sample_and_show_image(f"Result after epoch: {epoch}")
            self.view_reconstructed_images(x_0[0], 200)
            self.view_reconstructed_images(x_0[0], 500)
            self.view_reconstructed_images(x_0[0], 800)    

    def make_noisy_image(self, x, t):
        t = t.view(-1, 1)
        eps = torch.randn(x.shape)
        x_t = self.sqrt_alpha_hat[t] * x + self.sqrt_one_minus_alpha_hat[t] * eps
        
        return x_t, eps
    
    def sample_time_steps(self, n):
        return torch.randint(0, self.time_steps - 1, (n,))
    
    def save_model(self, filename):
        torch.save(self.state_dict(), filename)
        print("Model saved!")
        
    def load_model(self, filename):
        self.load_state_dict(torch.load(filename, map_location="cpu"))
        print("Model loaded!")
        
    def view_noisy_images(self, x, t, sizes):
        samples, _ = self.make_noisy_image(x,t)
        fig, axs = plt.subplots(len(t), 1)
        for sample, ax in zip(samples, axs):
            sample = torch.clamp(sample, 0, 1)
            ax.imshow(sample.view(*sizes))
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()
    
    def reconstruct_noisy_image(self, image, t):
        t = torch.tensor(t).view((1,))
        enc = self.time_encoding(t)
        noisy, _ = self.make_noisy_image(image.view((1,-1)), t)
        reconstructed = self(noisy.view(1,-1), enc)
        return noisy.detach(), reconstructed.detach()
    
    def view_reconstructed_images(self, x, t):
        noisy, reconstructed = self.reconstruct_noisy_image(x, t)
        noisy = torch.clamp(noisy, 0, 1)
        fig, axs = plt.subplots(1,3)
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
    
    def sample_image(self):
        x_t = torch.randn((1,self.img_size))
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