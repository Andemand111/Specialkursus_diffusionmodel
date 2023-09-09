from torch.utils.data import DataLoader
from model import Model
from data import dataset

dataloader = DataLoader(dataset, 
                        batch_size=64, 
                        shuffle=True, 
                        drop_last=True)

args = [dataset.size, 1000, 0.0001, 0.02]

model = Model(*args)
model.train(6, dataloader)
model.save_model("diffusion_model")
