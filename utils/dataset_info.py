import torch
from torchvision import transforms, datasets

class DataSet(object):

    def __init__(self, dataset_type, path="./data"):
        self.mean, self.std = self.get_mean_std(path, dataset_type)
        self.path = path
        self.dataset_type = dataset_type

    def get_data_set(self, path, data_set_type):
        simple_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
        if data_set_type == "mnist":
            return datasets.MNIST(path, train=True, download=True, transform=simple_transforms)
        elif data_set_type == "cifa":
            return datasets.CIFA(path, train=True, download=True, transform=simple_transforms)
        else:
            raise Exception("Unrecognized data set.")

    def get_mean_std(self, path, data_set_type, display_log=True):
        # simple transform
        exp = self.get_data_set(path, data_set_type)
        exp_data = exp.train_data
        exp_data = exp.transform(exp_data.numpy())
        self.mean = torch.mean(exp_data)
        self.std =  torch.std(exp_data)
        if display_log:
          print('[Train]')
          print(' - Numpy Shape:', exp.train_data.cpu().numpy().shape)
          print(' - Tensor Shape:', exp.train_data.size())
          print(' - min:', torch.min(exp_data))
          print(' - max:', torch.max(exp_data))
          print(' - mean:', self.mean)
          print(' - std:', self.std)
          print(' - var:', torch.var(exp_data))
        return self.mean, self.std

    def get_train_mnist_dataset(self, train_transform):
        train = datasets.MNIST(self.path, train=True, download=True, transform=train_transform)
        return train

    def get_test_mnist_dataset(self, test_transform):
        test = datasets.MNIST(self.path, train=False, download=True, transform=test_transform)
        return test

