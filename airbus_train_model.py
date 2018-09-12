
# coding: utf-8

# In[1]:


import torch
import torch.utils.data
import torchvision.datasets as datasets
import os
import struct
import errno
import numpy as np
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torchvision
import torchvision.transforms as transforms
from torch.utils import data as D
import torch.nn as nn
import torch.nn.functional as F
from torchnet import meter
import time

import airbus_dataloader
import airbus_train_val_functions


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*45*45, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*45*45)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN_fashion(nn.Module):
    def __init__(self, num_classes):
        super(CNN_fashion, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 5) # input features, output features, kernel size
        self.act1 = nn.ReLU()
        self.mp1 = nn.MaxPool2d(2, 2) # kernel size, stride
        
        self.conv2 = nn.Conv2d(32, 64, 5) # input features, output features, kernel size
        self.act2 = nn.ReLU()
        self.mp2 = nn.MaxPool2d(2, 2) # kernel size, stride
        
        self.fc = nn.Linear(64*45*45, num_classes) # 4x4 is the remaining spatial resolution here

    def forward(self, x):
        x = self.mp1(self.act1(self.conv1(x)))
        x = self.mp2(self.act2(self.conv2(x)))
        # The view flattens the output to a vector (the representation needed by the classifier)
        x = x.view(-1, 64*45*45)
        x = self.fc(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_gpu = torch.cuda.is_available()
batch_size = 8
workers = 4
path = '../airbus/'
aug=True
resize_factor=4
empty_frac=1
test_size=0.1
    
dataset = AirbusDS(torch.cuda.is_available(), batch_size, workers, path, aug, resize_factor, empty_frac, test_size)

# Define optimizer and loss function (criterion)
#model = CNN_fashion(2).to(device)
model = torch.load('CNN_fashion.model').to(device)
criterion = nn.CrossEntropyLoss().to(device)

# we can use advanced stochastic gradient descent algorithms 
# with regularization (weight-decay) or momentum
optimizer = torch.optim.SGD(model.parameters(), lr=0.001,
                            momentum=0.9,
                            weight_decay=5e-4)


total_epochs = 10
for epoch in range(total_epochs):
    print("EPOCH:", epoch + 1, time.strftime("%Y-%m-%d %H:%M:%S"))
    print("TRAIN")
    train(dataset.train_loader, model, criterion, optimizer, device)
    print("VALIDATION", time.strftime("%Y-%m-%d %H:%M:%S"))
    validate(dataset.val_loader, model, criterion, device)


