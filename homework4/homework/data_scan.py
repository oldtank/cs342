import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from .utils import DetectionSuperTuxDataset
from . import dense_transforms

if __name__ == '__main__':
    train_data = DetectionSuperTuxDataset('dense_data/train',
                                          transform=dense_transforms.Compose([
                                              dense_transforms.ToTensor(),
                                          dense_transforms.ToHeatmap()]))
    images = []
    ones=0
    zeros=0
    for img, peak, _ in train_data:
        images.append(img[None, :])
        ones+=torch.sum(peak==1)
        zeros+=torch.sum(peak==0)
    concat=torch.concat(images, dim=0)
    mean = torch.mean(concat, dim=[0,2,3])
    std = torch.std(concat, dim=[0,2,3])
    print("mean: " + repr(mean))
    print("std:  " + repr(std))
    print('ones: % 3d' % ones)
    print('zeros: % 3d' % zeros)

    # mean: tensor([0.2788, 0.2657, 0.2628])
    # std: tensor([0.2058, 0.1943, 0.2246])
    # ones: 7948
    # zeros: 334589755
