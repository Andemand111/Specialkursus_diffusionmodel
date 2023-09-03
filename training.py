from sklearn.datasets import fetch_lfw_people, load_digits
from torch.utils.data import DataLoader
import torch

from model import Model
from data import Data

data, _ = load_digits(return_X_y=True)
data = data / 15
h,b,c = 8,8,1

# data, _ = fetch_lfw_people(return_X_y = True, color=True)
# h, b, c = 62,47,3

dataset = Data(data)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

N = h*b*c
args = [N, 1000, 0.0001, 0.002]

model = Model(*args)
model.view_noisy_images(dataset[1].view(1,-1), torch.linspace(0,999,5).type(torch.int), [h,b,c])
model.train(150, dataloader)
model.save_model("diffusion_model_digits")

