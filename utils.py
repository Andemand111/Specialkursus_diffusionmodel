import torch
import torch.nn as nn

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
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 5, padding=2)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 5, padding=2)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 5, padding=2)
            self.transform = nn.Conv2d(out_ch, out_ch, 5, padding=2)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 5, padding=2)
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

class ResNET(nn.Module):
    def  __init__(self, channels=3, time_emb_dim=64):
        super().__init__()

        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)

        self.time_mlp = nn.Sequential(
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )
        
        self.conv0 = nn.Conv2d(channels, down_channels[0], 7, padding=3)

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

