from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

convert_tensor = transforms.ToTensor()

class Data(Dataset):
    def __init__(self, path, dimensions):
        super().__init__()
        self.path = path
        self.dimensions = dimensions
        
    def __len__(self):
        return 202598
    
    def __getitem__(self, index):
        index += 1
        n_zeros = 6 - len(str(index))
        n = n_zeros * "0" + str(index)
        img = Image.open(self.path + f'{n}.jpg')
        img  = convert_tensor(img).flatten()
        return img

path = "G:/Mit drev/Uni/5. semester/specialkursus/celeba/img_align_celeba/img_align_celeba/"
dataset = Data(path, [218, 178, 3])

pic = dataset[0]