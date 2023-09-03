from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import csv
import numpy as np

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']

class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path):
        self.data = []
        self.labels = []
        self.image_to_tensor = transforms.ToTensor()
        with open(dataset_path + '/labels.csv', "r") as file:
            reader = csv.reader(file)
            header = next(reader)
            self.dataset = {}
            for row in reader:
                image_array = np.array(Image.open(dataset_path + '/' + row[0]))
                self.data.append(self.image_to_tensor(image_array))
                self.labels.append(LABEL_NAMES.index(row[1]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
