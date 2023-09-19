import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from modules import ResUnet

class Model(nn.Module):
    def __init__(self, dimensions, time_steps, beta_start, beta_end):
        super().__init__()

        ## for the neural network
        self.dimensions = dimensions
        self.new_dimension = dimensions.copy()
        self.new_dimension[0] += 1

        self.img_size = torch.tensor(dimensions).prod()
        self.time_dim = dimensions[1] * dimensions[2]
        
        self.network = ResUnet(dimensions[0] + 1, dimensions[0])

        ## for noise scheduling
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
        x = torch.cat((x, t), dim=1)
        x = x.view(-1, *self.new_dimension)
        x = self.network(x)
        x = x.view(-1, self.img_size)
        return x


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
                loss = cost(pred, eps)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses[i] = loss.item()

            print(f"Epoch: {epoch}")
            print("loss: ", losses.mean().item())
            self.save_model(filename)

            self.sample_and_show_image(title=f"Result after epoch: {epoch}")

    def make_noisy_image(self, x_0, t):
        t = t.view(-1, 1)
        eps = torch.randn(x_0.shape)
        mu = self.sqrt_alpha_hat[t] * x_0
        std  = self.sqrt_one_minus_alpha_hat[t]
        x_t = mu + std * eps

        return x_t, eps

    def sample_time_steps(self, n):
        return torch.randint(self.time_steps, (n,))

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)
        print("Model saved!")

    def load_model(self, filename):
        self.load_state_dict(torch.load(filename, map_location="cpu"))
        print("Model loaded!")

    def get_prior_sample(self, x_t, t):
        with torch.no_grad():
            t_enc = self.time_encoding(torch.tensor(t).view((1,)))

            k1 = 1 / self.sqrt_alpha[t]
            k2 = (1 - self.alpha[t]) / self.sqrt_one_minus_alpha_hat[t]
            eps = self(x_t, t_enc) 
            x_t = k1 * (x_t - k2 * eps)

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

    def sample_and_show_image(self, x_T = None, title=""):
        x_t = self.sample_image(x_T)
        plt.imshow(x_t.view(*self.dimensions).permute(1, 2, 0), cmap="gray")
        plt.title(title)
        plt.axis("off")
        plt.show()

model = Model([3, 64, 64], 1000, 0.0001, 0.02)
dummy_input = torch.randn(2, 3 * 64 * 64)
dummy_time = model.sample_time_steps(2)
dummy_time_enc = model.time_encoding(dummy_time)
dummy_output = model(dummy_input, dummy_time_enc)
print("Testing output shape: ")
print(dummy_output.shape)