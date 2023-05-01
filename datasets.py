import glob
import random
import os
import torch.nn as nn
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from AffineT import *

class ImageDataset(Dataset):
    def __init__(self, root, lr_transforms=None, hr_transforms=None):
        self.srs_directory = os.path.join(root, 'SRS')
        self.srs_files = os.listdir(self.srs_directory)
        self.he_directory = os.path.join(root, 'HE')
        self.he_files = os.listdir(self.he_directory)

    def __getitem__(self, index):
        Pooling1 = nn.AvgPool2d(2, stride=2)
        
        # SRS images
        srs_filename = os.path.join(self.srs_directory, self.srs_files[index % len(self.srs_files)])
        srs_image = sio.loadmat(srs_filename)
        srs_image = srs_image['ss']

        # Normalizing
        srs_image[srs_image > 5] = 5
        srs_image[srs_image < 0] = 0
        srs_image = (srs_image - 2.5) / 2.5

        # H&E images
        he_filename = os.path.join(self.he_directory, self.srs_files[index % len(self.he_files)])
        he_image = sio.loadmat(he_filename)
        he_image = he_image['xx']

        lr_transforms = [transforms.ToTensor()]
        hr_transforms = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]

        self.lr_transform = transforms.Compose(lr_transforms)
        self.hr_transform = transforms.Compose(hr_transforms)
        
        # Apply random affine transformation
        shear = random.uniform(-0.15, 0.15)
        translation = (random.uniform(-0.15, 0.15), random.uniform(-0.15, 0.15))
        rotation = random.uniform(0, 180)
        scale = (random.uniform(0.85, 1.15), random.uniform(0.85, 1.15))
        affine_transform = Affine(rotation_range=rotation, translation_range=translation, shear_range=shear, zoom_range=scale)
        
        crop_start_x = random.randint(0, 114)
        crop_start_y = random.randint(0, 114)
        crop_size = 384

        image_name = self.srs_files[index % len(self.srs_files)]
        srs_image = affine_transform((self.lr_transform(srs_image)) + 2.5) - 2.5
        he_image = affine_transform((self.hr_transform(he_image)) - 1) + 1
        he_image = he_image[:, crop_start_x:crop_start_x + crop_size, crop_start_y:crop_start_y + crop_size]
        srs_image = srs_image[:, crop_start_x:crop_start_x + crop_size, crop_start_y:crop_start_y + crop_size]
        he_image[he_image > 1] = 1
        he_image[he_image < -1] = -1

        return {'srs': srs_image[1:, :, :], 'he': he_image, 'name': image_name}

    def __len__(self):
        return len(self.he_files)
