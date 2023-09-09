from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd

""" 
Download mnist som csv her: 
    https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
    
"""

class Data(Dataset):
    def __init__(self, data, dimensions):
        super().__init__()
        self.data = data
        self.dimensions = dimensions
        self.size = np.prod(dimensions)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        pic = self.data[index]
        return torch.from_numpy(pic).type(torch.float)
    
path = "../mnist_train.csv"
data = pd.read_csv(path).to_numpy()[:, 1:] / 255
dataset = Data(data, [28, 28, 1])