from torch.utils.data import Dataset
import numpy as np
import albumentations as A


# custom dataset class for albumentations library
class AlbumentationImageDataset(Dataset):
    def __init__(self, image_list, transforms=None):
        self.image_list = image_list
        self.transforms = transforms

    def __len__(self):
        return (len(self.image_list))

    def __getitem__(self, index):
        x, y = self.image_list[index]

        if self.transforms:
            print('Chosing transform ', y)
            x = self.transforms[y](x)  # Chose class transform based on y
        return x, y

    def __call__(self, image):
        img = np.array(image)
        return self.transforms(image=img)['image']


class Albumentation(object):
    """
  Helper class to create test and train transforms using Albumentations
  """

    def __init__(self, transforms=[]):
        self.transforms = A.Compose(transforms)

    def __call__(self, img):
        img = np.array(img)
        return self.transforms(image=img)['image']
