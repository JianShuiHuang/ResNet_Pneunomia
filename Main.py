# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 09:30:21 2020

@author: ivis
"""

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torchvision.models as models 
import pandas as pd
from torch.utils import data
from PIL import Image
from torchvision import transforms
from ResNet import *
from chest_xray_loader import *
from Time import *
import time
from TrainTest import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##Hyper parameters
lr = 1e-03
BatchSize = 4
Epochs18 = 10
Epochs50 = 5
Momentum = 0.9     
Weight_decay = 5e-4

def main():
    transformations = transforms.Compose([transforms.ToTensor()])
    train_data = DataLoader('train', transformations)
    test_data = DataLoader('test', transformations)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BatchSize, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=BatchSize, shuffle=False)  
    
    
    ##pretrain resnet18    
    train_accuracy = []
    test_accuracy = []
    
    model = Pretrained_model18()
    #model = torch.load('preresnet18.pkl').to(device)
    
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr = lr, momentum = Momentum, weight_decay = Weight_decay)
    
    start = time.time()

    for i in range(Epochs18):
        train = Train(train_dataloader, model, optimizer)
        train_accuracy.append(train)
        test = Test(test_dataloader, model, Epochs18)
        test_accuracy.append(test)
        print("epochs:", i )
        print('Train Accuracy: ', train)
        print('Test Accuracy: ', test)

    print("Time: ", timeSince(start, 1 / 100))

    print('Max accuracy: ', max(test_accuracy))
    print("resnet18 complet...")
    
    
    
    ##pretrain resnet50
    train_accuracy0 = []
    test_accuracy0 = []
    
    model0 = Pretrained_model50()
    #model0 = torch.load('preresnet50.pkl').to(device)
    
    model0 = model0.to(device)
    optimizer0 = optim.SGD(model0.parameters(), lr = lr, momentum = Momentum, weight_decay = Weight_decay)
    
    start = time.time()

    for i in range(Epochs50):
        train0 = Train(train_dataloader, model0, optimizer0)
        train_accuracy0.append(train0)
        test0 = Test(test_dataloader, model0, Epochs50)
        test_accuracy0.append(test0)
        print("epochs:", i )
        print('Train Accuracy: ', train0)
        print('Test Accuracy: ', test0)

    print("Time: ", timeSince(start, 1 / 100))

    print('Max accuracy: ', max(test_accuracy0))
    print("resnet50 complet...")
    

main()    
