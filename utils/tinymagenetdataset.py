from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from tqdm import notebook


class ProcessTinyImageNet(Dataset):

    path = "IMagenet/tiny-imagenet-200/"

    def __init__(self, classes_map):
        self.data = []
        self.target = []
        self.classes = classes_map
        self.update_train_val_data()

    @staticmethod
    def get_train_image_ids():
        """
            returns: ['n02321529',...]
        """
        path = ProcessTinyImageNet.path + "wnids.txt"
        train_image_ids = [line.strip() for line in open(path)]
        return train_image_ids

    @staticmethod
    def all_class_map():
        """
        eg:
        {'n02423022': 'label1'}
        """
        all_class_map = {}
        for line in open(ProcessTinyImageNet.path + "words.txt", 'r'):
            data = line.split('\t')[:2]
            n_id = data[0]
            class_name = data[1].strip()
            all_class_map[n_id] = class_name
        return all_class_map

    @staticmethod
    def get_image(image_path):
        img = Image.open(image_path)
        npimg = np.asarray(img)
        if len(npimg.shape) == 2:
            npimg = np.repeat(npimg[:, :, np.newaxis], 3, axis=2)
        return npimg

    def update_train_data(self):
        """
        returns: {'train/n02124075/images/n02124075_10.JPEG': 'Egyptian cat'}
        """
        image_ids = ProcessTinyImageNet.get_train_image_ids()
        class_map = self.classes
        train_image_path = ProcessTinyImageNet.path + "train/{image_id}/images/{image_name}.JPEG"
        for image_id in notebook.tqdm(image_ids, desc='Loading train folder...'):
            for key_id in range(500):
                image_name = image_id + "_" + str(key_id)
                train_path = train_image_path.format(image_id=image_id, image_name=image_name)
                npimg = ProcessTinyImageNet.get_image(train_path)
                train_label = class_map[image_id]
                self.data.append(npimg)
                self.target.append(train_label)

    def update_val_data(self):
        """
        returns: {'val/images/val_10.JPEG': 'swimming trunks, bathing trunks'}
        """
        path = ProcessTinyImageNet.path + "val/val_annotations.txt"
        class_map = self.classes
        val_image_path = ProcessTinyImageNet.path + "val/images/{image_name}"
        for line in notebook.tqdm(open(path, 'r'), desc='Loading val folder....'):
            data = line.split('\t')[:2]
            image_name = data[0].strip()
            image_id = data[1].strip()
            image_path = val_image_path.format(image_name=image_name)
            npimg = ProcessTinyImageNet.get_image(image_path)
            val_label = class_map[image_id]
            self.data.append(npimg)
            self.target.append(val_label)

    def update_train_val_data(self):
        self.update_train_data()
        self.update_val_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        target = self.target[idx]
        return data, target