# url
# load comlpeter data
# spilt data
# apply transform

# covert into dataloader

from torch.utils.data import Dataset, random_split
from PIL import Image
import numpy as np
import torch
import os
import torchvision.transforms as transforms
from tqdm import notebook
import zipfile
import requests
from io import StringIO, BytesIO
from helper import HelperModel

"""
This is used to download the tiny imagenet data set, load them, split to train test , convert to data set format
TinyImageNetDataSet - This is the main function which calls all all other. 
Parameters :
train_split : The percent of train data. Default it is 70%
test_transforms :Transformations to apply for test data
train_transforms : Transformations to apply for train data
Return Value : train_set, test_set of type dataset. Which are ready to go in Dataloader
Description of How it is implemented : 
TinyImageNetDataSet is the function which intern calls many funtions
1. Download_images - It dowloads the images from the given url and exact the zip file.
2. class_names - Derives the classes of tiny- Imagenet.
3. TinyImageNet - This returns the complete data of type data set.
4. Then we split the data we got from TinyImageNet class into train and test.
5. DatasetFromSubset - takes train or test data set and apply given transformations  
Finaly trainset, testset are returned.
"""


# -----------------------------------------------------Main Function which calls everything--------------------------------------------------------------
# def get_tinyimagenet_info(train_split=70, test_transforms=None, train_transforms=None):
#     down_url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
#     HelperModel.download_extract_images(down_url)
#     classes = class_names(url="tiny-imagenet-200/wnids.txt")
#     dataset = TinyImageNet(classes, url="tiny-imagenet-200")
#     train_len = len(dataset) * train_split // 100
#     test_len = len(dataset) - train_len
#     train_set, val_set = random_split(dataset, [train_len, test_len])
#     train_dataset = DatasetFromSubset(train_set, transform=train_transforms)
#     test_dataset = DatasetFromSubset(val_set, transform=test_transforms)
#
#     return train_dataset, test_dataset, classes


class ProcessTinyImagenet(object):
    def __init__(self, path, train_split=70):
        down_url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        HelperModel.download_extract_images(down_url)
        self.classes = class_names(url="tiny-imagenet-200/wnids.txt")
        dataset = TinyImageNet(self.classes, url=path)
        train_len = len(dataset) * train_split // 100
        test_len = len(dataset) - train_len
        self.train_set, self.val_set = random_split(dataset, [train_len, test_len])

    def train(self, train_transforms):
        train_dataset = DatasetFromSubset(self.train_set, transform=train_transforms)
        return train_dataset

    def test(self, test_transforms):
        test_dataset = DatasetFromSubset(self.val_set, transform=test_transforms)
        return test_dataset


# --------------------------------------------------------------Custom data set-------------------------------------------------------------------------

class TinyImageNet(Dataset):
    def __init__(self, classes, url):
        self.data = []
        self.target = []
        self.classes = classes
        self.url = url

        wnids = open(f"{url}/wnids.txt", "r")

        for wclass in notebook.tqdm(wnids, desc='Loading Train Folder', total=200):
            wclass = wclass.strip()
            for i in os.listdir(url + '/train/' + wclass + '/images/'):
                img = Image.open(url + "/train/" + wclass + "/images/" + i)
                npimg = np.asarray(img)

                if (len(npimg.shape) == 2):
                    npimg = np.repeat(npimg[:, :, np.newaxis], 3, axis=2)
                self.data.append(npimg)
                self.target.append(self.classes.index(wclass))

        val_file = open(f"{url}/val/val_annotations.txt", "r")
        for i in notebook.tqdm(val_file, desc='Loading Test Folder', total=10000):
            split_img, split_class = i.strip().split("\t")[:2]
            img = Image.open(f"{url}/val/images/{split_img}")
            npimg = np.asarray(img)
            if (len(npimg.shape) == 2):
                npimg = np.repeat(npimg[:, :, np.newaxis], 3, axis=2)
            self.data.append(npimg)
            self.target.append(self.classes.index(split_class))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        target = self.target[idx]
        img = data
        return data, target


# ----------------------------------------------------Data subset which comes after splitting--------------------------------------------------

class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


# -------------------------------------------------------------------------------------------------------------------------------------------------------

def class_names(url="tiny-imagenet-200/wnids.txt"):
    f = open(url, "r")
    classes = []
    for line in f:
        classes.append(line.strip())
    return classes
