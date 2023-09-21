from torch.utils.data import DataLoader
from model import Model
from data import Cifar10, Faces, MNIST

dataset = Faces()
dataloader = DataLoader(dataset, 
                        batch_size=64, 
                        shuffle=True, 
                        drop_last=True)

model = Model(dataset.dimensions, channels=3, time_steps=200)
filename = "diffusion_model"
model.train(30, dataloader, filename)