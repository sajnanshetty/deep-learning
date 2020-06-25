from torch.utils.data import Dataset
import numpy as np


class TorchVisionDataSet(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.dataset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.dataset)

    def __call__(self, image):
        img = np.array(image)
        return self.transforms(image=img)['image']
