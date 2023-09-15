from torch.utils.data import DataLoader
from model import Model
from data import Cifar10

dataset = Cifar10()
dataloader = DataLoader(dataset, 
                        batch_size=64, 
                        shuffle=True, 
                        drop_last=True)

args = [dataset.dimensions, 1000, 0.0001, 0.02]
model = Model(*args)
model.train(15, dataloader)
model.save_model("diffusion_model")
