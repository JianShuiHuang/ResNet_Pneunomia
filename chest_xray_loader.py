# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 21:25:44 2020

@author: ivis
"""
import torch
import numpy as np
import pandas as pd
from torch.utils import data
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def getData(mode):  
    if mode == 'train':
        train_dataset = ImageFolder(root="chest_xray/train/")
        
        train_data = []
        train_label = []
        
        for i in range(len(train_dataset)):
            train_data.append(train_dataset.imgs[i][0])
            train_label.append(train_dataset.imgs[i][1])        

        state = np.random.get_state()
        np.random.shuffle(train_data)
        np.random.set_state(state)
        np.random.shuffle(train_label)

        return np.squeeze(train_data), np.squeeze(train_label)
    
    else:
        test_dataset = ImageFolder(root="chest_xray/test/")
        
        test_data = []
        test_label = []
        
        for i in range(len(test_dataset)):
            test_data.append(test_dataset.imgs[i][0])
            test_label.append(test_dataset.imgs[i][1])

        state = np.random.get_state()
        np.random.shuffle(test_data)
        np.random.set_state(state)
        np.random.shuffle(test_label)     
             
        return np.squeeze(test_data), np.squeeze(test_label)
        

class DataLoader(data.Dataset):
    def __init__(self, mode, transform = None):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.img_name, self.label = getData(mode)
        self.mode = mode
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpeg rgb images during the training phase, such as rssh ubuntu@140.113.215.195 -p porttc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        ##step1
        path = self.img_name[index]
        img = Image.open(path)
        img = img.resize((64, 64),Image.ANTIALIAS)
        img = img.convert('RGB')
        
        ##step2
        GroundTruth = self.label[index]
        
        ##step3
        img_np = np.asarray(img)/255
        img_np = np.transpose(img_np, (2,0,1))
        img_ten = torch.from_numpy(img_np)
        
        ##step4
        return img_ten, GroundTruth

