import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Pretrained_model18():
    resnet18 = models.resnet18(pretrained = False)
    resnet18.fc = nn.Linear(in_features=512, out_features=5, bias=True)
    
    return resnet18

def Pretrained_model50():
    resnet50 = models.resnet50(pretrained = False)
    resnet50.fc = nn.Linear(in_features=2048, out_features=2, bias=True)
    
    return resnet50
