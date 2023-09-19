import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils import SuperTuxDataset

if __name__ == '__main__':
    train_data = SuperTuxDataset('../data/train')
    images = []
    for data, label in train_data:
        images.append(data[None, :])
    concat=torch.concat(images, dim=0)
    print('mean: ' + torch.mean(concat, dim=[0,2,3]))
    print('std: ' + torch.std(concat, dim=[0,2,3]))
