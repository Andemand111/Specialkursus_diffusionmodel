import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np

mse = nn.MSELoss()

def SinusoidalEmbeddings(ts):
    ## ts: (batch_size, 1)

    ## embeddings: (batch_size, 256)

    ## 256 dimensional embeddings used for the positional encoding

    time_dimension = 256
    half_dim =  time_dimension // 2
    embeddings = torch.log(torch.tensor(10000)) / (half_dim - 1)
    embeddings = torch.exp(torch.arange(half_dim).float() * -embeddings)
    embeddings = ts.view(-1, 1).float() * embeddings.view(1, -1)
    embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=1)

    return embeddings

class NoiseSchedule:
    ## Noise schedule for the noise added to the images
    ## beta_start: starting value for beta
    ## beta_end: ending value for beta
    ## time_steps: number of time steps
    ## the noise is then a linear interpolation between beta_start and beta_end such that 
    ## beta_0 = beta_start
    ## beta_T = beta_end
    ## beta_t = beta_start + (beta_end - beta_start) * t / T

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
        ## n: number of time steps to sample
        ## return timesteps t ~ U(1, T) (but zero indexed)

        return torch.randint(self.time_steps, (n,1))

    def make_noisy_images(self, x_0, t):
        ## x_0: initial image
        ## t: time steps to sample
        ## return x_t: noisy image
        ## return eps: noise
        ## return mu: mean of the distribution

        eps = torch.randn(x_0.shape)
        mu = self.sqrt_alpha_hat[t] * x_0
        std  = self.sqrt_one_minus_alpha_hat[t]
        x_t = mu + std * eps

        return x_t, eps, mu
    
class CIFAR10(Dataset):
    ## CIFAR10 dataset

    def __init__(self, n = None):
        ## n: number of images to use
        ## if n is None, use all images

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
    ## CelebA dataset

    def __init__(self, n = None, size=32):
        ## n: number of images to use
        ## if n is None, use all images
        ## size: size of the images

        super().__init__()

        self.n = n

        transform = transforms.Compose([
                transforms.CenterCrop(128),
                transforms.Resize(size),
                transforms.ToTensor(),
            ])

        self.images = torchvision.datasets.CelebA(root='../data', download=True, transform=transform)

    def __len__(self):
        if self.n is None:
            return len(self.images)
        
        return self.n
    
    def __getitem__(self, idx):
        ## return image in range [-1, 1]

        return self.images[idx][0] * 2 - 1
    
    
class Model(nn.Module):
    ## Base class for models

    def __init__(self, network, noise_schedule, dimensions, device):    
        ## network: network to use
        ## noise_schedule: noise schedule to use
        ## dimensions: dimensions of the images
        ## device: device to use (cpu or gpu)

        super().__init__()
        self.network = network.to(device)   
        self.noise_schedule = noise_schedule
        self.dimensions = dimensions
        self.img_size = torch.prod(torch.tensor(dimensions))
        self.device = device
    
    def sample(self, xt=None, T=None):
        ## xt: initial image
        ## T: number of time steps to sample
        ## return xt: sampled image
        ## return progress_list: list of sampled images at certain time steps

        ## the time steps at which progress is saved (to show reverse diffusion process)
        steps = torch.linspace(0, self.noise_schedule.time_steps, 10, dtype=torch.int)
        progress_list = []

        with torch.no_grad():
            xt = torch.randn((1, self.img_size)) if xt is None else xt  ## xt ~ N(0, I)
            T = self.noise_schedule.time_steps if T is None else T
            
            print("Sampling image..")
            for t in tqdm(reversed(range(0, T)), total = T - 1):
                torch.cuda.empty_cache() ## clear memory, otherwise it will crash due to the "big" loop
                
                xt = xt.to(self.device)

                xt = self.get_prior(xt, t)  ## find xt ~ p(x_{t-1} | x_t, x_0)
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
        ## x0: initial image
        ## ts: time steps to sample
        ## return loss: loss

        ## loss is calulated as:
        ## L_simple = E_{t ~ U(1, T)} [ || eps_true - eps_pred ||^2 ]

        batch_size = len(x0)

        ts = self.noise_schedule.sample_time_steps(batch_size) if ts is None else ts

        xt, eps, _ = self.noise_schedule.make_noisy_images(x0.flatten(1), ts)
        xt = xt.view(batch_size, *self.dimensions)

        xt = xt.to(self.device)
        eps = eps.to(self.device)
        ts = ts.to(self.device)

        pred = self(xt, ts.flatten())   ## predict eps = \hat{\eps}
        loss = mse(pred.flatten(1), eps)
        return loss
    
    def get_prior(self, xt, t):
        ## xt: image at time t
        ## t: time step
        ## return xt: image at time t - 1

        t_for_model = torch.tensor([t]).to(self.device)
        eps = self(xt.view(1, *self.dimensions), t_for_model).flatten()

        ## x_{t - 1} = k1 * (x_t - k2 * eps) + sigma * eps
        ## k1 = 1 / sqrt(alpha)
        ## k2 = (1 - alpha) / sqrt(1 - alpha_hat)

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
        ## x0: initial image
        ## ts: time steps to sample
        ## return loss: loss

        ## loss is calulated as:
        ## L_x0 = E_{t ~ U(1, T)} [ || x_0 - x_t ||^2 ]

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
        ## xt: image at time t
        ## t: time step
        ## return xt: image at time t - 1

        t_for_model = torch.tensor([t]).to(self.device)
        x0 = self(xt.view(1, *self.dimensions), t_for_model).flatten()

        if t == 0:
            return x0
        
        ## x_{t - 1} = k1 * (x_t - k2 * x_0) + sigma * eps
        
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


def test_model(model, dataset, batch_size = 64):
    ## model: model to test
    ## dataset: dataset to test on
    ## batch_size: batch size to use
    ## return test_loss: test loss

    model.eval()
    model.to(model.device)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loss = 0
    with torch.no_grad():
        for x in test_loader:
            loss = model.loss(x)
            test_loss += loss.item()
    test_loss /= len(test_loader)
    return test_loss

    
def show_losses(losses, test_loss = None):
    ## losses: losses to show
    ## test_loss: test loss to show

    train_losses = losses[:, 0]
    val_losses = losses[:, 1]
    
    if test_loss is not None:
        plt.scatter(len(losses) - 1, test_loss, label="Test loss", color='r', marker='x', s=100)

    plt.grid(True)
    plt.title("Losses") 
    plt.plot(train_losses, label="Train loss")
    plt.plot(val_losses, label="Validation loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.xticks(np.arange(0, len(losses)))
    plt.legend()
    plt.show()

def training_loop(model, epochs, train_set, val_set, batch_size=64, save_params=False):
    ## model: model to train
    ## epochs: number of epochs to train for
    ## train_set: training set
    ## val_set: validation set
    ## batch_size: batch size to use
    ## save_params: whether to save the parameters of the model at each epoch

    ## return epoch_loss: losses for each epoch
    ## return parameters: parameters for each epoch

    torch.autograd.set_detect_anomaly(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
    epoch_loss = torch.zeros((epochs, 2))

    num_paramters = sum(p.numel() for p in model.parameters() if p.requires_grad) 
    parameters = torch.zeros((epochs, num_paramters)) if save_params else None

    data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)

    for epoch in range(epochs):
        losses = torch.zeros(len(data_loader))

        for i, x0 in enumerate(tqdm(data_loader)):
            loss = model.loss(x0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses[i] = loss.item()
        
        model.save_model()
        
        train_loss = losses.mean().item()
        val_loss = test_model(model, val_set, batch_size)

        print(f"Epoch: {epoch}")
        print(f"Train loss: {train_loss}")
        print(f"Validation loss: {val_loss}")
        
        epoch_loss[epoch, 0] = train_loss
        epoch_loss[epoch, 1] = val_loss

        if save_params:
            ## save parameters if they have gradients
            curr_params = torch.concatenate([p.detach().cpu().flatten() for p in model.parameters() if p.requires_grad])
            parameters[epoch] = curr_params
    
    return epoch_loss, parameters

def sample_intermediate_images(model, title="Sampled images"):
    ## model: model to sample from
    ## title: title of the plot

    ## sample images during the reverse diffusion process

    model.eval()
    _, xts = model.sample()
    n_images = len(xts)
    fig, axs = plt.subplots(1, n_images, figsize=(20, 2))
    for i in range(n_images):
        img = xts[i] * 0.5 + 0.5
        img = torch.clamp(img, 0, 1)
        img = img.view(*model.dimensions).permute(1, 2, 0).cpu().numpy()
        axs[i].imshow(img, cmap='gray')
        axs[i].axis('off')
    fig.suptitle(title)
    plt.show()
    model.train()

def sample_grid(model, title="Sampled images"):
    ## model: model to sample from
    ## title: title of the plot

    ## sample 9 images from the model

    model.eval()
    print("Sampling 9 images for grid...")
    fig, axs = plt.subplots(3, 3, figsize=(8,8))
    for i in range(9):
        xt, _ = model.sample()
        xt = (xt + 1) / 2
        xt = torch.clamp(xt, 0, 1)
        img = xt.view(*model.dimensions).permute(1, 2, 0).cpu().numpy()
        axs[i//3, i%3].imshow(img, cmap='gray')
        axs[i//3, i%3].axis('off')
    fig.suptitle(title)
    plt.show()
    model.train()

def sample_approved_grid(model, title):
    ## model: model to sample from
    ## title: title of the plot

    ## sample 9 images from the model
    ## ask the user if the images are approved
    ## if not approved, sample another image

    model.eval()
    images = []
    i = 1
    while len(images) < 9:
        print(f"Sampling image {i}...")
        
        xt, _ = model.sample()
        xt = (xt + 1) / 2
        xt = torch.clamp(xt, 0, 1)
        img = xt.view(*model.dimensions).permute(1, 2, 0).cpu().numpy()
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Image {i}")
        plt.show()

        approved = input("Approve image? (y/n): ")
        if approved == 'y':
            images.append(img)
            i += 1
        elif approved == 'n':
            print("Image rejected")
        else:
            print("Invalid input")
        
        clear_output(wait=True)

    fig, axs = plt.subplots(3, 3, figsize=(8,8))
    for i in range(9):
        axs[i//3, i%3].imshow(images[i], cmap='gray')
        axs[i//3, i%3].axis('off')
    fig.suptitle(title)
    plt.show()