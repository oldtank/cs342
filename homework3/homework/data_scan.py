import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from .utils import SuperTuxDataset, DenseSuperTuxDataset

if __name__ == '__main__':
    train_data = SuperTuxDataset('data/train')
    images = []
    for data, label in train_data:
        images.append(data[None, :])
    concat=torch.concat(images, dim=0)
    mean = torch.mean(concat, dim=[0,2,3])
    std = torch.std(concat, dim=[0,2,3])
    print("mean: " + repr(mean))
    print("std:  " + repr(std))

    dense_train_data = DenseSuperTuxDataset('dense_data/train')
    images = []
    for data, label in dense_train_data:
        images.append(data[None, :])
    concat=torch.concat(images, dim=0)
    mean = torch.mean(concat, dim=[0,2,3])
    std = torch.std(concat, dim=[0,2,3])
    print("dense mean: " + repr(mean))
    print("dense std:  " + repr(std))