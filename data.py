from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import pickle
import torch
import matplotlib.pyplot as plt
import pandas as pd

class Faces(Dataset):
    def __init__(self):
        super().__init__()
        self.path = "G:/Mit drev/Uni/5. semester/specialkursus/celeba/img_align_celeba/img_align_celeba/"
        self.dimensions = [3, 64, 64]

    def convert_tensor(self, t):
        convert_tensor = transforms.ToTensor()
        return convert_tensor(t)

    def __len__(self):
        return 202599
    
    def __getitem__(self, index):
        index += 1
        n_zeros = 6 - len(str(index))
        n = n_zeros * "0" + str(index)
        img = Image.open(self.path + f'{n}.jpg')
        img  = self.convert_tensor(img)
        img = transforms.Resize(self.dimensions[1:], antialias=None)(img).flatten()
        return img

class MNIST(Dataset):
    def __init__(self):
        super().__init__()
        self.dimensions = [1, 28, 28]
        path = "G:/Mit drev/Uni/5. semester/specialkursus"
        self.data = pd.read_csv(path + "/mnist_train.csv").to_numpy()
        self.data = torch.tensor(self.data[:, 1:]) / 255
        self.data = self.data
    
    def __len__(self):
        return 60000
    
    def __getitem__(self, index):
        return self.data[index] * 2 - 1

class Cifar10(Dataset):
    def __init__(self):
        self.dimensions = [3, 32, 32]
        data = torch.zeros(50000, 3 * 32 * 32)
        labels = torch.zeros(50000)
        for i in range(1, 6):
            file  = f"C:/Users/Andba/Desktop/dtu/cifar-10-python/data_batch_{i}"
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            
            data[(i - 1) * 10000: i * 10000, :] = torch.tensor(dict[b"data"])
            labels[(i-1) * 10000: i * 10000] = torch.tensor(dict[b"labels"])

        self.labels = labels.long()
        self.data = data.float() / 255
        
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index].flatten(), self.labels[index]
    
