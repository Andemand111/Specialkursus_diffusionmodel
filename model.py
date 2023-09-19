import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn(self.conv1(x)))
        out = self.bn(self.conv2(out))
        if self.in_channels != self.out_channels:
            residual = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)(residual)
        
        out += residual  # Residual connection
        return out

class Model(nn.Module):
    def __init__(self, dimensions, time_steps, beta_start, beta_end):
        super().__init__()

        ## for the neural network
        self.dimensions = dimensions
        self.img_size = torch.tensor(dimensions).prod()
        self.time_dim = dimensions[1] * dimensions[2]

        num_res_blocks = 2
        in_channels = dimensions[0] + 1
        out_channels = dimensions[0]
        
        # Encoder (Downsampling)
        self.encoder = nn.ModuleList()
        for _ in range(num_res_blocks):
            self.encoder.append(ResidualBlock(in_channels, 64))
            in_channels = 64  # Update input channels for the next block
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck (Bottom of the U-Net)
        self.bottleneck = ResidualBlock(64, 64)

        # Decoder (Upsampling)
        self.decoder = nn.ModuleList()
        for _ in range(num_res_blocks):
            self.decoder.append(ResidualBlock(64, 64))

        # Final Convolution Layer (Output)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.linear = nn.Linear(self.img_size, self.img_size)

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
        x = torch.cat((x, t), 1)
        x = x.view(-1, self.dimensions[0] + 1, self.dimensions[1], self.dimensions[2])

        # Encoder
        encoder_outputs = []
        for block in self.encoder:
            x = block(x)
            encoder_outputs.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for block, skip_connection in zip(self.decoder, reversed(encoder_outputs)):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = block(x + skip_connection)

        # Final Convolution
        x = self.final_conv(x)
        x = x.view(-1, self.img_size)
        x = self.linear(x)

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

    def get_prior_sample(self, x_t, t, type="noise"):
        t_enc = self.time_encoding(torch.tensor(t).view((1,)))

        if type == "noise":
            k1 = 1 / self.sqrt_alpha[t]
            k2 = (1 - self.alpha[t]) / self.sqrt_one_minus_alpha_hat[t]
            eps = self(x_t, t_enc) 
            x_t = k1 * (x_t - k2 * eps)

        elif type == "x_0":
            k1 = self.sqrt_alpha[t] * self.one_minus_alpha_hat[t]
            k2 = self.sqrt_alpha_hat[t - 1] * (1 - self.alpha[t])
            x_0 = self(x_t, t_enc)
            x_t = (k1 * x_t + k2 * x_0) / self.one_minus_alpha_hat[t]

        if t > 1:
            x_t += torch.randn_like(x_t) * self.sigma[t]

        return x_t

    def sample_image(self, x_T = None, type="noise"):
        x_t = torch.randn((1, self.img_size)) if x_T == None else x_T
        for t in reversed(range(1, self.time_steps)):
            x_t = self.get_prior_sample(x_t, t, type)

        x_t = x_t * 0.5 + 0.5
        x_t = torch.clamp(x_t, 0, 1)
        return x_t.detach()

    def sample_and_show_image(self, x_T = None, type="noise", title=""):
        x_t = self.sample_image(x_T, type)
        plt.imshow(x_t.view(*self.dimensions).permute(1, 2, 0), cmap="gray")
        plt.title(title)
        plt.axis("off")
        plt.show()