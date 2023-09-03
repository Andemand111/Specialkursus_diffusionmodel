from torch.utils.data import Dataset
import torch


class Data(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        pic = self.data[index]
        return torch.from_numpy(pic).type(torch.float)