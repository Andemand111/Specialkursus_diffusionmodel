from model import UNetModel
from utils import SimpleModel, x0Model, NoiseSchedule
import matplotlib.pyplot as plt
import torch

""" first sample imagas from simple_model """

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
noise_schedule = NoiseSchedule(1e-4, 0.02, 1000)
network_args = {
    "in_channels": 3, 
    "model_channels": 64, 
    "out_channels": 3, 
    "num_res_blocks": 1, 
    "attention_resolutions": [2], 
    "dropout": 0.1,
    "num_heads": 8,
    "num_heads_upsample": 8,
}

simple_network = UNetModel(**network_args)
model_args = [simple_network, noise_schedule, [3,32,32], device]


simple_model = SimpleModel(*model_args)
simple_model.load_model()

num_images = 2

for i in range(num_images):
    xt, _ = simple_model.sample()
    xt = (xt + 1) / 2
    xt = torch.clamp(xt, 0, 1)
    img = xt.view(*simple_model.dimensions).permute(1, 2, 0).cpu().numpy()
    plt.imshow(img)
    plt.axis("off")
    plt.savefig(f"figures/simple_model_{i}.jpg", bbox_inches="tight", pad_inches=0)

del simple_network

""" now sample images from x0_model """

x0_model = x0Model(*model_args)
x0_model.load_model()

for i in range(num_images):
    xt, _ = x0_model.sample()
    xt = (xt + 1) / 2
    xt = torch.clamp(xt, 0, 1)
    img = xt.view(*simple_model.dimensions).permute(1, 2, 0).cpu().numpy()
    plt.imshow(img)
    plt.axis("off")
    plt.savefig(f"figures/x0_model_{i}.jpg", bbox_inches="tight", pad_inches=0)



