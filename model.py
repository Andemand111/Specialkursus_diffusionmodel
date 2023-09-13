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
        self.time_dim = 512
        
        """
        Der skal implementeres et netværk!
        """


        self.drop_out = nn.Dropout(p=0.1)
        
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
        """
        Her skal billedet køres igennem netværket!
        """

        return y
        
    def train(self, epochs, dataloader):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)
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
                print(loss.item())
                
                losses[i] = loss.item()
                
            print(f"Epoch: {epoch}")
            print(losses.mean().item())
            self.sample_and_show_image(f"Result after epoch: {epoch}")

        
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
        return reconstructed.detach()
    
    def sample_image(self):
        x_t = torch.randn((1,self.img_size))
        for t in reversed(range(1, self.time_steps)):
            t_enc = self.time_encoding(torch.tensor(t).view((1,)))
            x_0 = self(x_t, t_enc)
            k1 = self.sqrt_alpha[t] * self.one_minus_alpha_hat[t - 1] * x_t
            k2 = self.sqrt_alpha_hat[t - 1] * (1 - self.alpha[t]) * x_0
            k3 = self.one_minus_alpha_hat[t]
            k4 = self.sigma[t] * torch.randn_like(self.sigma[t])
            x_t = (k1 + k2) / k3 + k4
        return x_t.detach()
    
    def sample_and_show_image(self, title=""):
        x_t = self.sample_image()
        plt.imshow(x_t.view(*self.dimensions), cmap="gray")
        plt.title(title)
        plt.show()