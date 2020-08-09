from torchvision.transforms import transforms
import torch
from torch.utils.data import Dataset
from PIL import Image
from utils.helper import get_stastics_map
import os


class CustomDataset(Dataset):
    def __init__(self, image_path, start=1, end=100):
        self.bg_image_list = []
        self.fg_bg_image_list = []
        self.dense_image_list = []
        self.mask_image_list = []
        print("*****************start*****************")
        for bg_number in range(start, end + 1):
            index_no = (bg_number - 1) * 4000 + 1
            for count in range(1, 4001):
                sub_image_path = f'bg_{bg_number:03d}/{index_no}.jpg'
                self.bg_image_list.append(os.path.join(image_path, "background", f'{bg_number}.jpg'))
                self.fg_bg_image_list.append(os.path.join(image_path, "fg_bg", sub_image_path))
                self.dense_image_list.append(os.path.join(image_path, "fg_bg_dense_depth", sub_image_path))
                self.mask_image_list.append(os.path.join(image_path, "fg_bg_mask", sub_image_path))
                index_no += 1
        print("*****************end*****************")

    def __getitem__(self, idx):
        fg_bg = self.fg_bg_image_list[idx]
        mask = self.mask_image_list[idx]
        dense = self.dense_image_list[idx]
        bg = self.bg_image_list[idx]
        return {"fg_bg": fg_bg, "bg": bg, "mask": mask, "dense": dense}

    def __len__(self):
        return len(self.fg_bg_image_list)


def apply_transform(image_type, resize_image=128, transform=None):
    image_type = get_stastics_map()[image_type]
    mean = image_type["mean"]
    std = image_type["std"]
    apply_transform = [
        transforms.Resize((resize_image, resize_image)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),

    ]
    if transform:
        transform = transform + apply_transform
    transform = apply_transform
    return transforms.Compose(transform)


class SubDatasetTransform(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, custmadataset, resize_image=128, transform=None):
        """
        Args:
            data (string): zipped images and labels.
        """
        self.custmset = custmadataset
        self.bg_transform = apply_transform('bg', resize_image)
        self.fg_bg_transform = apply_transform('fg_bg', resize_image)
        self.mask_transform = apply_transform('mask', resize_image)
        self.dense_transform = apply_transform('dense', resize_image)

    def __len__(self):
        return len(self.custmset)

    def __getitem__(self, idx):
        image_map = self.custmset[idx]
        bg = Image.open(image_map["bg"])
        fg_bg = Image.open(image_map["fg_bg"])
        mask = Image.open(image_map["mask"]).convert("L")
        dense = Image.open(image_map["dense"]).convert("L")
        bg_transformed = self.bg_transform(bg)
        fg_bg_transformed = self.fg_bg_transform(fg_bg)
        mask_transformed = self.mask_transform(mask)
        dense_transformed = self.dense_transform(dense)
        input_images = torch.cat((bg_transformed, fg_bg_transformed), dim=0)
        return {"input_images": input_images, "bg": bg_transformed, "fg_bg": fg_bg_transformed,
                "mask": mask_transformed, "dense": dense_transformed}
