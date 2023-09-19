from torch.utils.data import DataLoader
from model import Model
from data import Cifar10, Faces, MNIST

dataset = MNIST()
dataloader = DataLoader(dataset, 
                        batch_size=128, 
                        shuffle=True, 
                        drop_last=True)

args = [dataset.dimensions, 1000, 0.0001, 0.02]
model = Model(*args)
filename = "diffusion_model"
model.train(30, dataloader, filename)