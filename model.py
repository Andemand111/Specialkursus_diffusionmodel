import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

class Model(nn.Module):
    def __init__(self, img_size, time_steps, beta_start, beta_end):
        super().__init__()
        
        self.img_size = img_size
        time_dim = int(img_size/4)
        self.time_dim = time_dim if time_dim % 2 == 0 else time_dim + 1
        
        
        n_nodes = [128, 64, 32]
        self.lin1 = nn.Linear(self.img_size + self.time_dim, n_nodes[0])
        self.lin2 = nn.Linear(n_nodes[0], n_nodes[1])
        self.lin3 = nn.Linear(n_nodes[1], n_nodes[2])
        self.lin4 = nn.Linear(n_nodes[2], n_nodes[1])
        self.lin5 = nn.Linear(n_nodes[1], n_nodes[0])
        self.lin6 = nn.Linear(n_nodes[0], self.img_size)
        
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
        frequencies = torch.exp(torch.arange(0, self.time_dim, 2).float() * -(math.log(10000.0) / self.time_dim))
        angles = t.unsqueeze(1) * frequencies.unsqueeze(0)
        encodings = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        
        return encodings     
    
    def forward(self, x, t):
        y = torch.cat((x, t), dim=1)
        y = F.relu(self.lin1(y))
        y = F.relu(self.lin2(y))
        y = F.relu(self.lin3(y))
        y = F.relu(self.lin4(y))
        y = F.relu(self.lin5(y))
        y = F.sigmoid(self.lin6(y))
        
        return y
        
    def train(self, epochs, dataloader):
        optimizer = torch.optim.Adam(self.parameters())
        mse = nn.MSELoss()
        batch_size = dataloader.batch_size
        
        for epoch in range(epochs):
            losses = torch.zeros(len(dataloader))
            
            for i, x in enumerate(tqdm(dataloader)):
                ts = self.sample_time_steps(batch_size)
                x_T, eps = self.make_noisy_image(x, ts)
                time_encodings = self.time_encoding(ts)
                pred = self(x_T, time_encodings)
                loss = mse(pred, x)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses[i] = loss.item()
                
            print(f"Epoch: {epoch}")
            print(losses.mean().item())
        
    def make_noisy_image(self, x, t):
        t = t.view(-1, 1)
        eps = torch.randn((len(t), x.shape[1])) 
        sample = self.sqrt_alpha_hat[t] * x + self.sqrt_one_minus_alpha_hat[t] * eps
        
        return sample, eps
    
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
    
    def reconstruc_noisy_image(self, image, t):
        t = torch.tensor(t).view((1,))
        enc = self.time_encoding(t)
        noisy, _ = self.make_noisy_image(image.view((1,-1)), t)
        reconstructed = self(noisy.view(1,-1), enc)
        return reconstructed.detach()
        