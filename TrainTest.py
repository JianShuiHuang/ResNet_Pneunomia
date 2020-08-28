import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
import time
from Time import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Train(data, model, optimizer):
    ##training
    model.train()
    true = 0
    false = 0
    Loss = nn.CrossEntropyLoss()   #change loss function here
    
    for i, data_ten in enumerate(data):
        train_data, train_label = data_ten
        train_data, train_label = train_data.to(device), train_label.to(device)
            
        x_train = Variable(train_data)
        y_train = Variable(train_label)
            
        prediction = model(x_train.float())
        
        guess = torch.max(prediction, 1)[1]
        for j in range(len(guess)):
            if guess[j] == train_label[j]:
                true = true + 1
            else:
                false = false + 1
        
        if i % 200 == 0:
            print(i, " true: ", true, " false: ", false)
            
        loss = Loss(prediction, y_train.long())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    if true / (true + false) > 0.7:
        torch.save(model, 'preresnet18.pkl')

    return true / (true + false)

def Test(data, model, epoch):
    ##testing
    model.eval()
    true = 0
    false = 0
    
    for i, data_ten in enumerate(data):
        test_data, test_label = data_ten
        test_data, test_label = test_data.to(device), test_label.to(device)
            
        x_test = Variable(test_data)
            
        prediction = model(x_test.float())
        
        guess = torch.max(prediction, 1)[1]
             
        for j in range(len(guess)):
            if guess[j] == test_label[j]:
                true = true + 1
            else:
                false = false + 1
        
        if i % 200 == 0:
            print(i, " true: ", true, " false: ", false)

    return true / (true + false)
