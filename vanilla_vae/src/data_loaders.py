import os
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.datasets import MNIST
from torchvision.io import read_image
from torchvision import transforms
from typing import Optional


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        # download only
        MNIST(os.getcwd(), train=True, download=True, transform=self.transform)

    def setup(self, stage: Optional[str] = None):
        # transform
        mnist_train = MNIST(os.getcwd(), train=True, download=False, transform=self.transform)

        # train/val split
        mnist_train, mnist_val = random_split(mnist_train, [55000, 5000])

        # assign to use in dataloaders
        self.train_dataset = mnist_train
        self.val_dataset = mnist_val

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)


class CelebADataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class CropCelebA64(object):
    """ This class applies cropping for CelebA64. This is a simplified implementation of:
    https://github.com/andersbll/autoencoding_beyond_pixels/blob/master/dataset/celeba.py
    """

    def __call__(self, pic):
        new_pic = pic.crop((15, 40, 178 - 15, 218 - 30))
        return new_pic

    def __repr__(self):
        return self.__class__.__name__ + '()'


class CelebADataModule(pl.LightningDataModule):
    def __init__(self, train_batch_size=64, val_batch_size=64):
        super().__init__()

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            CropCelebA64(),
            transforms.Resize((64, 64)),
            transforms.ToTensor()])

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        dataset = CelebADataset(annotations_file='E:/Datasets/CelebA/list_attr_celeba.csv',
                                img_dir='E:/Datasets/CelebA/img_align_celeba/img_align_celeba',
                                transform=self.transform)

        train, val = random_split(dataset, [162079, 40520])

        self.train_dataset = train
        self.val_dataset = val

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size)
