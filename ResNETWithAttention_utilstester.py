#%%
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm

mse = nn.MSELoss()

def SinusoidalEmbeddings(ts):
    time_dimension = 64
    half_dim =  time_dimension // 2
    embeddings = torch.log(torch.tensor(10000)) / (half_dim - 1)
    embeddings = torch.exp(torch.arange(half_dim).float() * -embeddings)
    embeddings = ts.view(-1, 1).float() * embeddings.view(1, -1)
    embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=1)

    return embeddings

class NoiseSchedule:
    def __init__(self, beta_start, beta_end, time_steps):
        super().__init__()

        self.beta_start = beta_start
        self.beta_end = beta_end
        self.time_steps = time_steps

        self.beta = torch.linspace(self.beta_start, self.beta_end, self.time_steps)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha = torch.sqrt(self.alpha)
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        self.one_minus_alpha_hat = 1 - self.alpha_hat
        self.sqrt_one_minus_alpha_hat = torch.sqrt(self.one_minus_alpha_hat)
        self.sigma_sq = (1 - self.alpha) * torch.roll(self.one_minus_alpha_hat, 1) / self.one_minus_alpha_hat
        self.sigma_sq[0] = 0
        self.sigma = torch.sqrt(self.sigma_sq)

    def sample_time_steps(self, n):
        return torch.randint(self.time_steps, (n,1))

    def make_noisy_images(self, x_0, t):
        eps = torch.randn(x_0.shape)
        mu = self.sqrt_alpha_hat[t] * x_0
        std  = self.sqrt_one_minus_alpha_hat[t]
        x_t = mu + std * eps

        return x_t, eps, mu

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, out_ch)
        )
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 3, padding=1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
        
    def forward(self, x, t, ):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb.view(-1, time_emb.shape[1], 1, 1)
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)
    
    # Attention block

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        
        # Query, Key, Value projections
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        
        # Output projection
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        batch_size, C, width, height = x.size()
        
        # Linear projections
        Q = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        K = self.key(x).view(batch_size, -1, width * height)
        V = self.value(x).view(batch_size, -1, width * height)
        
        # Attention Score
        attention = torch.bmm(Q, K.permute(0, 2, 1))
        attention = self.softmax(attention)
        
        # Get weighted features
        out = torch.bmm(V, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        
        out = self.gamma * out + x
        return out
    
    # Incorporating attention block into the Block "block".
class BlockWithAttention(Block):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super(BlockWithAttention, self).__init__(in_ch, out_ch, time_emb_dim, up)
        
        self.attention = SelfAttention(out_ch)
        
    def forward(self, x, t):
        x = super().forward(x, t)
        x = self.attention(x)
        return x



class ResNET(nn.Module):
    def  __init__(self, channels=3, time_emb_dim=64, down_channels = [64, 128, 256, 512, 1024]):
        super().__init__()

        up_channels = down_channels[::-1]

        self.time_mlp = nn.Sequential(
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )
        
        self.conv0 = nn.Conv2d(channels, down_channels[0], 3, padding=1)

        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], \
                                    time_emb_dim) \
                    for i in range(len(down_channels)-1)])

        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                        time_emb_dim, up=True) \
                    for i in range(len(up_channels)-1)])
        
        self.output = nn.Sequential(
            nn.Conv2d(up_channels[-1], channels, 3, padding=1),
        )

    def forward(self, x, t):
        t = self.time_mlp(t)
        x = self.conv0(x)
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1)           
            x = up(x, t)

        out = self.output(x)

        return out
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
        print("Model saved to {}".format(path))

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        print("Model loaded from {}".format(path))


# Modified ResNET with attention blocks
class ResNETWithAttention(ResNET):
    def __init__(self, channels=3, time_emb_dim=64, down_channels=[64, 128, 256, 512, 1024]):
        super().__init__(channels, time_emb_dim, down_channels)
        
        up_channels = down_channels[::-1]
        
        self.downs = nn.ModuleList([BlockWithAttention(down_channels[i], down_channels[i+1], time_emb_dim) for i in range(len(down_channels)-1)])
        self.ups = nn.ModuleList([BlockWithAttention(up_channels[i], up_channels[i+1], time_emb_dim, up=True) for i in range(len(up_channels)-1)])


class MNIST(Dataset):
    def __init__(self, n = None):
        super().__init__()

        self.n = n
        data = torchvision.datasets.MNIST(root='../data', download=True, transform=transforms.ToTensor())
        data = data.data.float() / 255
        data = data.view(-1, 1, 28, 28)
        self.data = data * 2 - 1

    def __len__(self):
        if self.n is None:
            return len(self.data)
        
        return self.n
    
    def __getitem__(self, idx):
        return self.data[idx]
    
class CIFAR10(Dataset):
    def __init__(self, n = None):
        super().__init__()

        data = torchvision.datasets.CIFAR10(root='../data', download=True, transform=transforms.ToTensor())
        data = data.data.float() / 255
        data = data.view(-1, 3, 32, 32)
        self.data = data * 2 - 1

    def __len__(self):
        if self.n is None:
            return len(self.data)
        
        return self.n
    
    def __getitem__(self, idx):
        return self.data[idx]

class SimpleModel(nn.Module):
    def __init__(self, network, train_loader, noise_schedule, dimensions, device):
        super().__init__()
        self.network = network
        self.train_loader = train_loader
        self.noise_schedule = noise_schedule
        self.dimensions = dimensions
        self.img_size = torch.prod(torch.tensor(dimensions))
        self.device = device
        self.path = "C:/Users/gusta/OneDrive/Skrivebord/KI & Data/Semester 5/Diffusionsprojekt/simple_model.pt"
    
    def loss(self, x0):
        ts = self.noise_schedule.sample_time_steps(self.train_loader.batch_size)
        embeddings = SinusoidalEmbeddings(ts)
        xt, eps, _ = self.noise_schedule.make_noisy_images(x0.flatten(1), ts)
        xt = xt.view(self.train_loader.batch_size, *self.dimensions)

        xt = xt.to(self.device)
        embeddings = embeddings.to(self.device)
        eps = eps.to(self.device)

        pred = self.network(xt, embeddings)
        loss = mse(pred.flatten(1), eps)
        return loss
    
    def sample(self):
        with torch.no_grad():
            xt = torch.randn((1, self.img_size))
            print("Sampling image..")
            for t in tqdm(reversed(range(1, self.noise_schedule.time_steps)), total=self.noise_schedule.time_steps-1):
                torch.cuda.empty_cache() ## clear memory, otherwise it will crash due to the "big" loop
                embedding = SinusoidalEmbeddings(torch.tensor(t).view(1,1))
                
                xt = xt.to(self.device)
                embedding = embedding.to(self.device)

                eps = self.network(xt.view(1, *self.dimensions), embedding).flatten()

                k1 = 1 / self.noise_schedule.sqrt_alpha[t]
                k2 = (1 - self.noise_schedule.alpha[t]) / self.noise_schedule.sqrt_one_minus_alpha_hat[t]

                xt = k1 * (xt - k2 * eps)
                if t > 1:
                    xt += self.noise_schedule.sigma[t].to(self.device) * torch.randn((1, self.img_size)).to(self.device)

            print("Done sampling image")

            return xt
        
    def save_model(self):
        torch.save(self.state_dict(), self.path)
        print("Model saved to {}".format(self.path))
    
    def load_model(self):
        try:
            self.load_state_dict(torch.load(self.path))
            print("Model loaded from {}".format(self.path))
        except:
            print("Failed to load model from {}".format(self.path))
            print("Initializing new model")
    

class MuModel(nn.Module):
    def __init__(self, network, train_loader, noise_schedule, dimensions, device):
        super().__init__()
        self.network = network
        self.train_loader = train_loader
        self.noise_schedule = noise_schedule
        self.dimensions = dimensions
        self.img_size = torch.prod(torch.tensor(dimensions))
        self.device = device
        self.path = "../models/mu_model.pt"
    
    def loss(self, x0):
        ts = self.noise_schedule.sample_time_steps(self.train_loader.batch_size)
        embeddings = SinusoidalEmbeddings(ts)
        xt, _, _ = self.noise_schedule.make_noisy_images(x0.flatten(1), ts)
        xt = xt.view(self.train_loader.batch_size, *self.dimensions)

        xt = xt.to(self.device)
        embeddings = embeddings.to(self.device)
        x0 = x0.to(self.device)

        pred = self.network(xt, embeddings)
        loss = mse(pred.flatten(1), x0.flatten(1))
        return loss
    
    def sample(self):
        with torch.no_grad():
            xt = torch.randn((1, self.img_size))
            print("Sampling image..")
            for t in tqdm(reversed(range(1, self.noise_schedule.time_steps)), total=self.noise_schedule.time_steps-1):
                torch.cuda.empty_cache() ## clear memory, otherwise it will crash due to the "big" loop
                embedding = SinusoidalEmbeddings(torch.tensor(t).view(1,1))
                
                xt = xt.to(self.device)
                embedding = embedding.to(self.device)

                x0 = self.network(xt.view(1, *self.dimensions), embedding).flatten(1)
                k1 = self.noise_schedule.sqrt_alpha[t] * (1 - self.noise_schedule.alpha_hat[t - 1]) * xt
                k2 = self.noise_schedule.sqrt_alpha_hat[t - 1] * (1 - self.noise_schedule.alpha[t]) * x0
                xt = (k1 + k2) / (1 - self.noise_schedule.alpha_hat[t])

                if t > 1:
                    xt += self.noise_schedule.sigma[t].to(self.device) * torch.randn((1, self.img_size)).to(self.device)

            print("Done sampling image")

            return xt
        
    def save_model(self):
        torch.save(self.state_dict(), self.path)
        print("Model saved to {}".format(self.path))
    
    def load_model(self):
        try:
            self.load_state_dict(torch.load(self.path))
            print("Model loaded from {}".format(self.path))
        except:
            print("Failed to load model from {}".format(self.path))
            print("Initializing new model")
# %%
