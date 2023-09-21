import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F

from . import dense_transforms

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']
DENSE_LABEL_NAMES = ['background', 'kart', 'track', 'bomb/projectile', 'pickup/nitro']
# Distribution of classes on dense training set (background and track dominate (96%)
DENSE_CLASS_DISTRIBUTION = [0.52683655, 0.02929112, 0.4352989, 0.0044619, 0.00411153]


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path, resize=None, random_crop=None, flip=False, color_aug=False, random_erase=False):
        """
        Your code here
        Hint: Use the python csv library to parse labels.csv
        """
        import csv
        from os import path
        self.data = []

        #transforms to apply to the image
        tf = []
        # if resize is not None:
        #     tf.append(transforms.Resize(resize))

        # if random_crop is not None:
        #     # tf.append(transforms.Resize(resize))
        #     tf.append(transforms.RandomCrop(random_crop))

        if flip:
            tf.append(transforms.RandomHorizontalFlip())

        if color_aug:
            tf.append(transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3))
            tf.append(transforms.RandomGrayscale(p=0.4)) 
            
        tf.append(transforms.ToTensor())
        # tf.append(transforms.Normalize(mean=[0.3234, 0.3310, 0.3444], std=[0.2524, 0.2219, 0.2470]))
        if random_erase:
            tf.append(transforms.RandomErasing(p=0.5, scale=(0.02,0.33), ratio=(0.3,3.3)))   
            
        transform_to_perform = transforms.Compose(tf)

        with open(path.join(dataset_path, 'labels.csv'), newline='') as f:
            reader = csv.reader(f)
            for fname, label, _ in reader:
                if label in LABEL_NAMES:
                    image = Image.open(path.join(dataset_path, fname))
                    label_id = LABEL_NAMES.index(label)
                    self.data.append((transform_to_perform(image), label_id))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Your code here
        return a tuple: img, label
        """
        return self.data[idx]


class DenseSuperTuxDataset(Dataset):
    def __init__(self, dataset_path, transform=dense_transforms.ToTensor()):
        from glob import glob
        from os import path
        self.files = []
        for im_f in glob(path.join(dataset_path, '*_im.jpg')):
            self.files.append(im_f.replace('_im.jpg', ''))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        b = self.files[idx]
        im = Image.open(b + '_im.jpg')
        lbl = Image.open(b + '_seg.png')
        if self.transform is not None:
            im, lbl = self.transform(im, lbl)
        # print(im.shape)
        return im, lbl


def load_data(dataset_path, num_workers=0, batch_size=128, **kwargs):
    dataset = SuperTuxDataset(dataset_path, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def load_dense_data(dataset_path, num_workers=0, batch_size=32, flip=False, color_aug=False, crop=False):
    tf = []

    if flip:
        tf.append(dense_transforms.RandomHorizontalFlip())

    if color_aug:
        tf.append(dense_transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3))

    if crop:
        # tf.append(dense_transforms.RandomCrop(size=100))
        tf.append(dense_transforms.Resize((128, 144)))
        tf.append(dense_transforms.RandomResizedCrop(size=(96, 128)))

    tf.append(dense_transforms.ToTensor())

    dataset = DenseSuperTuxDataset(dataset_path, transform=dense_transforms.Compose(tf))
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def _one_hot(x, n):
    return (x.view(-1, 1) == torch.arange(n, dtype=x.dtype, device=x.device)).int()


class ConfusionMatrix(object):
    def _make(self, preds, labels):
        label_range = torch.arange(self.size, device=preds.device)[None, :]
        preds_one_hot, labels_one_hot = _one_hot(preds, self.size), _one_hot(labels, self.size)
        return (labels_one_hot[:, :, None] * preds_one_hot[:, None, :]).sum(dim=0).detach()

    def __init__(self, size=5):
        """
        This class builds and updates a confusion matrix.
        :param size: the number of classes to consider
        """
        self.matrix = torch.zeros(size, size)
        self.size = size

    def add(self, preds, labels):
        """
        Updates the confusion matrix using predictions `preds` (e.g. logit.argmax(1)) and ground truth `labels`
        """
        self.matrix = self.matrix.to(preds.device)
        self.matrix += self._make(preds, labels).float()

    @property
    def class_iou(self):
        true_pos = self.matrix.diagonal()
        return true_pos / (self.matrix.sum(0) + self.matrix.sum(1) - true_pos + 1e-5)

    @property
    def iou(self):
        return self.class_iou.mean()

    @property
    def global_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos.sum() / (self.matrix.sum() + 1e-5)

    @property
    def class_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos / (self.matrix.sum(1) + 1e-5)

    @property
    def average_accuracy(self):
        return self.class_accuracy.mean()

    @property
    def per_class(self):
        return self.matrix / (self.matrix.sum(1, keepdims=True) + 1e-5)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()

if __name__ == '__main__':
    dataset = DenseSuperTuxDataset('dense_data/train', transform=dense_transforms.Compose(
        [dense_transforms.RandomHorizontalFlip(), dense_transforms.ToTensor()]))
    from pylab import show, imshow, subplot, axis

    for i in range(15):
        im, lbl = dataset[i]
        subplot(5, 6, 2 * i + 1)
        imshow(F.to_pil_image(im))
        axis('off')
        subplot(5, 6, 2 * i + 2)
        imshow(dense_transforms.label_to_pil_image(lbl))
        axis('off')
    show()
    import numpy as np

    c = np.zeros(5)
    for im, lbl in dataset:
        c += np.bincount(lbl.view(-1), minlength=len(DENSE_LABEL_NAMES))
    print(100 * c / np.sum(c))
