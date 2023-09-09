import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

class Model(nn.Module):
    def __init__(self, img_size, time_steps, beta_start, beta_end):
        super().__init__()
        
        self.img_size = img_size
        self.time_dim = 128
        
        self.conv1 = nn.Conv2d(1, 8, (4,4), (2,2))
        self.conv2 = nn.Conv2d(8, 16, (3,3))
        self.conv3 = nn.Conv2d(16, 32, (3, 3))
        self.conv4 = nn.Conv2d(32, 64, (3,3))
        self.conv5 = nn.Conv2d(64, 128, (3,3))
        self.lin1 = nn.Linear(3328, 4096)
        self.lin2 = nn.Linear(4096, 2048)
        self.lin3 = nn.Linear(2048, 1024)
        self.lin4 = nn.Linear(1024, 784)
        
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
        
        self.beta_tilde = (1 - torch.roll(self.alpha_hat, 1)) / (1 - self.alpha_hat) * self.beta
        self.beta_tilde[0] = 0
        
    def time_encoding(self, t):
        frequencies = torch.arange(0, self.time_dim, 2) * torch.pi / self.time_steps
        angles = t.unsqueeze(1) * frequencies.unsqueeze(0)
        encodings = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        
        return encodings     
    
    def forward(self, x, t):
        x = x.view(-1, 28, 28, 1).permute(0, 3, 1, 2)
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = F.relu(self.conv4(y))
        y = F.relu(self.conv5(y))
        
        y = torch.flatten(y, 1)
        y = torch.cat((y, t), 1)
        
        y = F.relu(self.lin1(y))
        y = F.relu(self.lin2(y))
        y = F.relu(self.lin3(y))
        y = F.tanh(self.lin4(y))
        y = (y + 1) / 2

        return y
        
    def train(self, epochs, dataloader):
        optimizer = torch.optim.Adam(self.parameters())
        mse = nn.MSELoss()
        batch_size = dataloader.batch_size
        
        for epoch in range(epochs):
            losses = torch.zeros(len(dataloader))
            
            for i, x_0 in enumerate(tqdm(dataloader)):
                ts = self.sample_time_steps(batch_size)
                x_t, eps = self.make_noisy_image(x_0, ts)
                time_encodings = self.time_encoding(ts)
                pred = self(x_t, time_encodings)
                loss = mse(pred, x_0)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses[i] = loss.item()
                
            print(f"Epoch: {epoch}")
            print(losses.mean().item())
        
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
        