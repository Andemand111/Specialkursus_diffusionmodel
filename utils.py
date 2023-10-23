import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm

mse = nn.MSELoss()


def SinusoidalEmbeddings(ts):
    time_dimension = 256
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
        data = torch.from_numpy(data.data)
        data = data.permute(0,3,1,2).float() / 255
        self.data = data * 2 - 1
        self.n = n

    def __len__(self):
        if self.n is None:
            return len(self.data)
        
        return self.n
    
    def __getitem__(self, idx):
        return self.data[idx]
    
class CelebA(Dataset):
    def __init__(self, n = None, size=32):
        super().__init__()

        self.n = n

        try:
            self.data = torch.load(f"../data/celeba{size}.pt")
        except:
            print("Could not load celeba.pt, downloading data..")
            transform = transforms.Compose([
                transforms.CenterCrop(128),
                transforms.Resize(size),
                transforms.ToTensor(),
            ])

            self.images = torchvision.datasets.CelebA(root='../data', download=True, transform=transform)

            num_images = 100_000
            self.data = torch.zeros((num_images, 3, size, size))

            print("Loading images into tensor..")
            for i in tqdm(range(num_images)):
                self.data[i] = self.images[i][0] * 2 - 1

            torch.save(self.data, f"../data/celeba{size}.pt")

    def __len__(self):
        if self.n is None:
            return len(self.data)
        
        return self.n
    
    def __getitem__(self, idx):
        return self.data[idx]


class StanfordCars(Dataset):
    def __init__(self, n = None):
        super().__init__()

        data = torchvision.datasets.StanfordCars(root='../data', download=True, transform=transforms.ToTensor())
        data = torch.stack(data.data)
        data = data.permute(0,3,1,2).float() / 255
        self.data = data * 2 - 1
        self.n = n

    def __len__(self):
        if self.n is None:
            return len(self.data)
        
        return self.n
    
    def __getitem__(self, idx):
        return self.data[idx]
    
class Model(nn.Module):
    def __init__(self, network, noise_schedule, dimensions, device):
        super().__init__()
        self.network = network.to(device)   
        self.noise_schedule = noise_schedule
        self.dimensions = dimensions
        self.img_size = torch.prod(torch.tensor(dimensions))
        self.device = device
    
    def sample(self, xt=None, T=None):
        steps = torch.linspace(0, self.noise_schedule.time_steps, 10, dtype=torch.int)
        progress_list = []

        with torch.no_grad():
            xt = torch.randn((1, self.img_size)) if xt is None else xt
            T = self.noise_schedule.time_steps if T is None else T
            
            print("Sampling image..")
            for t in tqdm(reversed(range(0, T)), total = T - 1):
                torch.cuda.empty_cache() ## clear memory, otherwise it will crash due to the "big" loop
                
                xt = xt.to(self.device)

                xt = self.get_prior(xt, t)
                if t in steps:
                    progress_list.append(xt.view(*self.dimensions).cpu())

            print("Done sampling image")

            return xt, progress_list
        
    def save_model(self, path= None):
        if path is not None:
            self.path = path

        torch.save(self.state_dict(), self.path)
        print("Model saved to {}".format(self.path))
    
    def load_model(self, path= None):
        if path is not None:
            self.path = path
            
        try:
            self.load_state_dict(torch.load(self.path))
            print("Model loaded from {}".format(self.path))
        except:
            print("Failed to load model from {}".format(self.path))
            print("Initializing new model")


class SimpleModel(Model):
    def __init__(self, network, noise_schedule, dimensions, device):
        super().__init__(network, noise_schedule, dimensions, device)
        self.path = "../models/simple_model.pt"
    
    def loss(self, x0, ts= None):
        batch_size = len(x0)

        ts = self.noise_schedule.sample_time_steps(batch_size) if ts is None else ts

        xt, eps, _ = self.noise_schedule.make_noisy_images(x0.flatten(1), ts)
        xt = xt.view(batch_size, *self.dimensions)

        xt = xt.to(self.device)
        eps = eps.to(self.device)
        ts = ts.to(self.device)

        pred = self(xt, ts.flatten())
        loss = mse(pred.flatten(1), eps)
        return loss
    
    def get_prior(self, xt, t):
        t_for_model = torch.tensor([t]).to(self.device)
        eps = self(xt.view(1, *self.dimensions), t_for_model).flatten()

        k1 = 1 / self.noise_schedule.sqrt_alpha[t]
        k2 = (1 - self.noise_schedule.alpha[t]) / self.noise_schedule.sqrt_one_minus_alpha_hat[t]

        xt = k1 * (xt - k2 * eps)
        if t > 0:
            xt += self.noise_schedule.sigma[t].to(self.device) * torch.randn((1, self.img_size)).to(self.device)
        
        return xt
    
    def forward(self, x, t):
        return self.network(x, t)
    

class x0Model(Model):
    def __init__(self, network, noise_schedule, dimensions, device):
        super().__init__(network, noise_schedule, dimensions, device)
        self.path = "../models/x0_model.pt"
    
    def loss(self, x0, ts= None):
        batch_size = len(x0)
        ts = self.noise_schedule.sample_time_steps(batch_size) if ts is None else ts

        xt, _, _ = self.noise_schedule.make_noisy_images(x0.flatten(1), ts)
        xt = xt.view(batch_size, *self.dimensions)

        xt = xt.to(self.device)
        x0 = x0.to(self.device)
        ts = ts.to(self.device)

        pred = self(xt, ts.flatten())
        loss = mse(pred.flatten(1), x0.flatten(1))
        return loss
    
    def get_prior(self, xt, t):
        t_for_model = torch.tensor([t]).to(self.device)
        x0 = self(xt.view(1, *self.dimensions), t_for_model).flatten()

        if t == 0:
            return x0
        
        k1 = self.noise_schedule.sqrt_alpha[t] * (1 - self.noise_schedule.alpha_hat[t - 1]) * xt
        k2 = self.noise_schedule.sqrt_alpha_hat[t - 1] * (1 - self.noise_schedule.alpha[t]) * x0
        xt = (k1 + k2) / (1 - self.noise_schedule.alpha_hat[t])
        xt += self.noise_schedule.sigma[t].to(self.device) * torch.randn((1, self.img_size)).to(self.device)

        return xt

    def forward(self, x, t):
        return torch.tanh(self.network(x, t))
    
class ELBOModel(Model):
    def __init__(self, network, noise_schedule, dimensions, device):
        super().__init__(network, noise_schedule, dimensions, device)
        self.path = "../models/elbo_model.pt"
    
    def loss(self, x0):
        ts = torch.arange(0, self.noise_schedule.time_steps).view(-1,1)
        embeddings = SinusoidalEmbeddings(ts)
        xts, _, _ = self.noise_schedule.make_noisy_images(x0.flatten(1).repeat(len(ts), 1), ts)

        reconstruction_loss = self.reconstruction_loss(x0, xts[1])
        kl_loss = self.kl_loss(xts, embeddings)

        loss = reconstruction_loss + kl_loss

        return loss

    def kl_loss(self, xts, embeddings):
        kl_loss = 0

        for t in range(1, self.noise_schedule.time_steps):
            constant = 1 / (2 * self.noise_schedule.beta[t])

            xt_plus_one = xts[t + 1].view(1, *self.dimensions).to(self.device)
            embedding = embeddings[t + 1].view(1, -1).to(self.device)

            xt_minus_one = xts[t - 1].view(1, *self.dimensions)

            mu_pred = self.network(xt_plus_one, embedding)
            mu_real = self.noise_schedule.sqrt_alpha[t] * xt_minus_one

            kl_loss += constant * mse(mu_pred.flatten(1), mu_real.flatten(1))
        
        return kl_loss

    def reconstruction_loss(self, x0, x1):
        constant = 1 / (2 * self.noise_schedule.beta[0])
        embedding = SinusoidalEmbeddings(torch.tensor(1).view(1,1))
        embedding = embedding.to(self.device)
        x0 = x0.to(self.device)
        x1 = x1.to(self.device)

        mu_pred = self.network(x0, embedding)
        loss = constant * mse(mu_pred.flatten(1), x1.flatten(1))
        return loss
    
    def get_prior(self, xt, embedding, t):
        xt = self.network(xt.view(1, *self.dimensions), embedding).flatten(1)

        if t > 0:
            xt += self.noise_schedule.beta[t] * torch.randn((1, self.img_size)).to(self.device)
