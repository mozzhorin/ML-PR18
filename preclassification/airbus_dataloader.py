'''
Preclassification: Data load & data preparation
'''
# coding: utf-8

import numpy as np
import pandas as pd
from PIL import Image
import os

from sklearn.model_selection import train_test_split

import torch
import torch.utils.data
from torch.utils import data as D
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms




# <api>
# https://github.com/albu/albumentations
# Used for the augmentation
from albumentations import (ToFloat, Resize,
    CLAHE, RandomRotate90, Transpose, ShiftScaleRotate, Blur, OpticalDistortion, 
    GridDistortion, HueSaturationValue, IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, 
    MedianBlur, IAAPiecewiseAffine, IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, 
    Flip, OneOf, Compose
)



# <api>
# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape=(768, 768)):
    '''
    Decode a mask from RLE to numpy array.
        mask_rle: run-length as string formated (start length)
        shape: (height,width) of array to return 
        Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction





class AirbusDS_train(D.Dataset):
    """
    A customized data loader for train dataset.
    """
    def __init__(self, path_train, aug, transform, ids, masks):
        """ Intialize the dataset.
        """
        self.aug = aug
        self.path_train = path_train
        self.transform = transform
        self.df = ids
        self.masks = masks       
        self.filenames = self.df['ImageId'].values
        self.len = len(self.filenames)
        
    
    def get_mask(self, ImageId):
        img_masks = self.masks.loc[self.masks['ImageId'] == ImageId, 'EncodedPixels'].tolist()

        # Take the individual ship masks and create a single mask array for all ships
        all_masks = np.zeros((768, 768))
        if img_masks == [-1]:
            return all_masks
        for mask in img_masks:
            all_masks += rle_decode(mask)
        return all_masks
    
    def get_label(self, ImageId):
        '''Returns a label: 0 - no ship, 1 - has one or more ships.'''
        label = int(self.df[self.df['ImageId']==ImageId]['has_ship'].values[0])
        return label
    
    def __getitem__(self, index):
        """ Get a sample from the dataset.
        """       
        
        image = Image.open(str(self.path_train + self.filenames[index]))
        ImageId = self.filenames[index]
        label = self.get_label(ImageId)
        mask = self.get_mask(ImageId)
        # Check if augmentation is required.
        if self.aug:
            data = {"image": np.array(image), "mask": mask}
            transformed = self.transform(**data)
            image = transformed['image']/255
            image = np.transpose(image, (2, 0, 1))
            return image, transformed['mask'][np.newaxis,:,:], label
        else:        
            return self.transform(image), mask[np.newaxis,:,:], label 

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len



# <api>
class AirbusDS_val(AirbusDS_train):
    """
    A customized data loader for validation during training.
    """
    
    def __getitem__(self, index):
        """ Get a sample from the dataset
        """       
        
        image = Image.open(str(self.path_train + self.filenames[index]))
        ImageId = self.filenames[index]
        label = self.get_label(ImageId)
                    
        if self.aug:
            data = {"image": np.array(image)}
            transformed = self.transform(**data)
            image = transformed['image']/255
            image = np.transpose(image, (2, 0, 1))
            return image, label
        else:
            return self.transform(image), label


class AirbusDS_test(D.Dataset):
    """
    A customized data loader for test data set (during predictions for submission).
    """
    def __init__(self, path_test, transform):
        """ Intialize the dataset
        """
        
        self.path_test = path_test
        self.transform = transform
        self.test_files = np.array(os.listdir(self.path_test))
        self.len = len(self.test_files)
        
    def __getitem__(self, index):
        """ Get a sample from the dataset
        """       
        
        image = Image.open(str(self.path_test + self.test_files[index]))
        ImageId = self.test_files[index]
        return self.transform(image), ImageId
    
    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len


class AirbusDS:
    """
    A wrapper around train/val/test data sets to produce them together.

    Parameters:
        atch_size (int) : batch size.
        workers(int) : number of processes.
        is_gpu (bool): True if CUDA is enabled. 
        root (str): root folder, contains train & test subfolders with data sets.
        aug (bool): augmentation during training is required (True) or not (False).
        resize_factor (int): resize factor for images. 
        empty_frac (float from (0., 1.)): the ratio of empty images used by training (1 = all empty images).  
        test_size (float from (0., 1.)): the size of validation data set used by training (1 = 100%).
        

    Attributes:
        trainset (torch.utils.data.TensorDataset): Training set wrapper.
        valset (torch.utils.data.TensorDataset): Validation set wrapper.
        train_loader (torch.utils.data.DataLoader): Training set loader with shuffling.
        val_loader (torch.utils.data.DataLoader): Validation set loader.
    """

    def __init__(self, is_gpu, batch_size, workers, root, aug=False, resize_factor=1, empty_frac=0.33, test_size=0.1):
        
        self.root = root
        self.path_train = root + 'train/'
        self.path_test = root + 'test/'
        self.aug = aug
        self.empty_frac = empty_frac
        self.resize_factor = resize_factor
        self.test_size = test_size
        
        # List of corrupted images 
        exclude_list = ['6384c3e78.jpg','13703f040.jpg', '14715c06d.jpg',  '33e0ff2d5.jpg',
                '4d4e09f2a.jpg', '877691df8.jpg', '8b909bb20.jpg', 'a8d99130e.jpg', 
                'ad55c3143.jpg', 'c8260c541.jpg', 'd6c7f17c7.jpg', 'dc3e7c901.jpg',
                'e44dffe88.jpg', 'ef87bad36.jpg', 'f083256d8.jpg']   
    
        # Calculate masks
        masks = pd.read_csv(str(self.root+'train_ship_segmentations.csv')).fillna(-1)
        masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
        
        # Calculate the number of ships on the images
        unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
        unique_img_ids['has_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x>0 else 0.0)
        unique_img_ids['has_ship_vec'] = unique_img_ids['has_ship'].map(lambda x: [x])
        
        # Drop corrupted images
        unique_img_ids = unique_img_ids[~unique_img_ids['ImageId'].isin(exclude_list)]
        self.images_df = unique_img_ids
        
        masks.drop(['ships'], axis=1, inplace=True)
        self.masks = masks 
        
        ## Create data sets/loaders
        self.trainset, self.valset, self.testset = self.get_dataset()
        self.train_loader, self.val_loader, self.test_loader = self.get_dataset_loader (batch_size, workers, is_gpu)
        self.val_loader.dataset.class_to_idx = {'no_ship': 0, 'ship': 1}


    def get_dataset(self):
        """
        Loads and wraps training and validation datasets

        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """
        
        # Split dataset to train and validate sets to evaluate the model
        train_ids, val_ids = train_test_split(self.images_df, test_size=self.test_size)
        self.val_ids = val_ids
        
        # Drop small images (mostly just clouds, water or corrupted)
        train_ids['file_size_kb'] = train_ids['ImageId'].map(lambda c_img_id: os.stat(os.path.join(self.path_train, c_img_id)).st_size/1024)
        train_ids = train_ids[train_ids['file_size_kb']>40] # keep only >40kb files
        
        # Undersample empty images to balance the dataset
        ships = train_ids[train_ids['has_ship']==1] 
        no_ships = train_ids[train_ids['has_ship']==0].sample(frac=self.empty_frac)  # Take only this fraction of empty images
        self.train_ids = pd.concat([ships, no_ships], axis=0)  
        
        # Define transformations for augmentation and without it
        self.transform_no_aug = transforms.Compose([transforms.Resize((int(768/self.resize_factor),
                                                                       int(768/self.resize_factor))), transforms.ToTensor()])
        if self.aug:
            self.transform = Compose([Resize(height=int(768/self.resize_factor), width=int(768/self.resize_factor)),
                                      OneOf([RandomRotate90(), Transpose(), Flip()], p=0.3)])
        else:
            self.transform = self.transform_no_aug

        # TensorDataset wrapper
        trainset = AirbusDS_train(self.path_train, self.aug, self.transform, self.train_ids, self.masks)
        valset = AirbusDS_val(self.path_train, False, self.transform_no_aug, self.val_ids, self.masks) 
        testset = AirbusDS_test(self.path_test, self.transform)

        return trainset, valset, testset

    def get_dataset_loader(self, batch_size, workers, is_gpu):
        """
        Defines the dataset loader for wrapped dataset

        Parameters:
            batch_size (int): Defines the batch size in data loader
            workers (int): Number of parallel threads to be used by data loader
            is_gpu (bool): True if CUDA is enabled so pin_memory is set to True

        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True,
                                                   num_workers=workers, pin_memory=is_gpu, sampler=None)
        val_loader = torch.utils.data.DataLoader(self.valset, batch_size=batch_size, shuffle=True,
                                                  num_workers=workers, pin_memory=is_gpu, sampler=None)
        test_loader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=True,
                                                  num_workers=workers, pin_memory=is_gpu, sampler=None)

        return train_loader, val_loader, test_loader
