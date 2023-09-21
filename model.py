import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from modules import (
    ResidualConv,
    ASPP,
    AttentionBlock,
    Upsample_,
    Squeeze_Excite_Block,
)

class Model(nn.Module):
    def __init__(self, dimensions, channels = 3, filters=[32, 64, 128, 256, 512], time_steps=1000):
        super().__init__()

        ## for the neural network
        self.dimensions = dimensions
        self.img_size = torch.tensor(dimensions).prod()
        self.new_dimension = dimensions.copy()
        self.new_dimension[0] += 1
        self.time_dim = dimensions[1] * dimensions[2]

        self.input_layer = nn.Sequential(
            nn.Conv2d(channels + 1, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channels + 1, filters[0], kernel_size=3, padding=1)
        )

        self.squeeze_excite1 = Squeeze_Excite_Block(filters[0])

        self.residual_conv1 = ResidualConv(filters[0], filters[1], 2, 1)

        self.squeeze_excite2 = Squeeze_Excite_Block(filters[1])

        self.residual_conv2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.squeeze_excite3 = Squeeze_Excite_Block(filters[2])

        self.residual_conv3 = ResidualConv(filters[2], filters[3], 2, 1)

        self.aspp_bridge = ASPP(filters[3], filters[4])

        self.attn1 = AttentionBlock(filters[2], filters[4], filters[4])
        self.upsample1 = Upsample_(2)
        self.up_residual_conv1 = ResidualConv(filters[4] + filters[2], filters[3], 1, 1)

        self.attn2 = AttentionBlock(filters[1], filters[3], filters[3])
        self.upsample2 = Upsample_(2)
        self.up_residual_conv2 = ResidualConv(filters[3] + filters[1], filters[2], 1, 1)

        self.attn3 = AttentionBlock(filters[0], filters[2], filters[2])
        self.upsample3 = Upsample_(2)
        self.up_residual_conv3 = ResidualConv(filters[2] + filters[0], filters[1], 1, 1)

        self.aspp_out = ASPP(filters[1], filters[0])

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(3, 1, kernel_size=3, padding=1),
            nn.Flatten(),
            nn.Linear(self.dimensions[1] * self.dimensions[2], self.img_size),
            nn.Tanh(),
            )

        ## for noise scheduling
        self.time_steps = time_steps
        self.beta_start = 1e-4
        self.beta_end = 0.02

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

    def time_encoding(self, t):
        frequencies = torch.arange(0, self.time_dim, 2) * torch.pi / self.time_steps
        angles = t.unsqueeze(1) * frequencies.unsqueeze(0)
        encodings = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

        return encodings

    def forward(self, x, t):
        x = torch.cat((x, t), dim=1)
        x = x.view(-1, *self.new_dimension)

        x1 = self.input_layer(x) + self.input_skip(x)

        x2 = self.squeeze_excite1(x1)
        x2 = self.residual_conv1(x2)

        x3 = self.squeeze_excite2(x2)
        x3 = self.residual_conv2(x3)

        x4 = self.squeeze_excite3(x3)
        x4 = self.residual_conv3(x4)

        x5 = self.aspp_bridge(x4)

        x6 = self.attn1(x3, x5)
        x6 = self.upsample1(x6)
        x6 = torch.cat([x6, x3], dim=1)
        x6 = self.up_residual_conv1(x6)

        x7 = self.attn2(x2, x6)
        x7 = self.upsample2(x7)
        x7 = torch.cat([x7, x2], dim=1)
        x7 = self.up_residual_conv2(x7)

        x8 = self.attn3(x1, x7)
        x8 = self.upsample3(x8)
        x8 = torch.cat([x8, x1], dim=1)
        x8 = self.up_residual_conv3(x8)

        out = self.aspp_out(x8)
        out = self.output_layer(out)

        return out


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

            #k1 = 1 / self.sqrt_alpha[t]
            #k2 = (1 - self.alpha[t]) / self.sqrt_one_minus_alpha_hat[t]
            #x_t = k1 * (x_t - k2 * self(x_t, t_enc))

            k1 = self.sqrt_alpha[t] * self.one_minus_alpha_hat[t - 1] * x_t
            k2 = self.sqrt_alpha_hat[t - 1] * (1 - self.alpha[t]) * self(x_t, t_enc)
            x_t = (k1 + k2) / self.one_minus_alpha_hat[t]

            if t > 1:
                x_t += torch.randn_like(x_t) * self.sigma[t]

            return x_t

    def sample_image(self, x_T = None):
        print("Sampling image...")
        x_t = torch.randn((1, self.img_size)) if x_T == None else x_T
        for t in reversed(range(1, self.time_steps)):
            x_t = self.get_prior_sample(x_t, t)

        x_t = x_t * 0.5 + 0.5
        x_t = torch.clamp(x_t, 0, 1)
        print("Done!")
        return x_t.detach()

    def sample_and_show_image(self, x_T = None, title=""):
        x_t = self.sample_image(x_T)
        plt.imshow(x_t.view(*self.dimensions).permute(1, 2, 0), cmap="gray")
        plt.title(title)
        plt.axis("off")
        plt.show()

model = Model([3, 64, 64], channels=3)
dummy_input = torch.randn(2, 3 * 64 * 64)
dummy_time = model.sample_time_steps(2)
dummy_time_enc = model.time_encoding(dummy_time)
dummy_output = model(dummy_input, dummy_time_enc)
print("Testing output shape: ")
print(dummy_output.shape)
print("Should be 2 x (3 * 64 * 64)")