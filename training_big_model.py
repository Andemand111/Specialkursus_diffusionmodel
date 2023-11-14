from utils import CelebA, NoiseSchedule, training_loop, SimpleModel
from model import UNetModel
from torch.utils.data import random_split
import torch

data = CelebA(size = 32)
torch.seed = 42

split_fracs = [0.8, 0.15, 0.05]
train_set, test_set, val_set = random_split(data, split_fracs)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dimensions = train_set[0].shape
img_size = torch.prod(torch.tensor(dimensions))

noise_schedule = NoiseSchedule(1e-4, 0.02, 1000)

network_args = {
    "in_channels": dimensions[0], 
    "model_channels": 06, 
    "out_channels": dimensions[0], 
    "num_res_blocks": 2, 
    "attention_resolutions": [2], 
    "dropout": 0.1,
    "num_heads": 16,
    "num_heads_upsample": 16,
}

big_network = UNetModel(**network_args)
model_args = [big_network, noise_schedule, dimensions, device]
big_model = SimpleModel(*model_args)
big_model.path = "../models/big_model.pt"
big_model.load_model()
num_params = sum(p.numel() for p in big_model.parameters() if p.requires_grad)
print("Number of parameters: ", num_params)

losses, _ = training_loop(
    model = big_model, 
    epochs = 30, 
    train_set = train_set, 
    val_set = val_set,  
    batch_size = 64, 
    save_params=False)

torch.save(losses, "../models/big_model_losses.pt")