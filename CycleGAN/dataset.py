import os
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import cv2

import matplotlib.pyplot as plt
import numpy as np

import glob
import warnings
warnings.filterwarnings("ignore")
def get_data_loader(data_path, image_size=512, batch_size=16, num_workers=0):  #128
    """Returns training and test data loaders for a given image type, either 'summer' or 'winter'. 
       These images will be resized to 128x128x3, by default, converted into Tensors, and normalized.
    """
    
    # resize and normalize the images
    train_transform = transforms.Compose([transforms.Resize((image_size,image_size)), # resize to 128x128
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.Resize((image_size,image_size)), # resize to 128x128
                                    transforms.ToTensor()])

    # get training and test directories
    
    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'valid')

    # define datasets using ImageFolder
    train_dataset = datasets.ImageFolder(train_path, train_transform)
    test_dataset = datasets.ImageFolder(test_path, test_transform)

    # create and return DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader