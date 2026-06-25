import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import Subset, DataLoader
from torchvision.transforms import AutoAugmentPolicy
from PIL import Image
from pathlib import Path
import numpy as np
import os

def CIFAR10(root = "./data", download = False, val = False, batch_size=128, config = None):
    transform_aug = []

    train_transform = transforms.Compose(
        transform_aug + [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if val :
        np.random.seed(0)
        split_permutation = list(np.random.permutation(50000))

        train_set = Subset(torchvision.datasets.CIFAR10(root = root, train=True, transform = train_transform, download=download), split_permutation[:45000])
        val_set = Subset(torchvision.datasets.CIFAR10(root = root, train=True, transform = test_transform, download=download), split_permutation[45000:])
        test_set = torchvision.datasets.CIFAR10(root = root , train=False, transform = test_transform, download=download)
    
        train_loader = DataLoader(train_set,batch_size=batch_size,  pin_memory=True, shuffle = True)
        val_loader   = DataLoader(val_set,batch_size=batch_size,  pin_memory=True, shuffle = False)
        test_loader   = DataLoader(test_set,batch_size=batch_size,  pin_memory=True, shuffle = False)


        return train_loader, val_loader, test_loader
    else:
        train_set = torchvision.datasets.CIFAR10(root = root, train=True, transform = train_transform, download=download)
        test_set = torchvision.datasets.CIFAR10(root = root , train=False, transform = test_transform, download=download)
    
        train_loader = DataLoader(train_set,batch_size=batch_size,  pin_memory=True, shuffle = True)
        val_loader   = None
        test_loader   = DataLoader(test_set,batch_size=batch_size,  pin_memory=True, shuffle = False)


        return train_loader, val_loader, test_loader
