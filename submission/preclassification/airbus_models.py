'''
Preclassification: simple CNN models.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_32_x4(nn.Module):
    def __init__(self, num_classes):
        super(CNN_32_x4, self).__init__()
        
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
    
class CNN_8_x4(nn.Module):
    def __init__(self, num_classes):
        super(CNN_8_x4, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 8, 5) # input features, output features, kernel size
        self.act1 = nn.ReLU()
        self.mp1 = nn.MaxPool2d(2, 2) # kernel size, stride
        
        self.conv2 = nn.Conv2d(8, 16, 5) # input features, output features, kernel size
        self.act2 = nn.ReLU()
        self.mp2 = nn.MaxPool2d(2, 2) # kernel size, stride
        
        self.fc = nn.Linear(16*45*45, num_classes) # 4x4 is the remaining spatial resolution here

    def forward(self, x):
        x = self.mp1(self.act1(self.conv1(x)))
        x = self.mp2(self.act2(self.conv2(x)))
        # The view flattens the output to a vector (the representation needed by the classifier)
        x = x.view(-1, 16*45*45)
        x = self.fc(x)
        return x
    
class CNN_32_x2(nn.Module):
    def __init__(self, num_classes):
        super(CNN_32_x2, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 5) # input features, output features, kernel size
        self.act1 = nn.ReLU()
        self.mp1 = nn.MaxPool2d(2, 2) # kernel size, stride
        
        self.conv2 = nn.Conv2d(32, 64, 5) # input features, output features, kernel size
        self.act2 = nn.ReLU()
        self.mp2 = nn.MaxPool2d(2, 2) # kernel size, stride
        
        self.fc = nn.Linear(64*93*93, num_classes) # 4x4 is the remaining spatial resolution here

    def forward(self, x):
        x = self.mp1(self.act1(self.conv1(x)))
        x = self.mp2(self.act2(self.conv2(x)))
        # The view flattens the output to a vector (the representation needed by the classifier)
        x = x.view(-1, 64*93*93)
        x = self.fc(x)
        return x
    
class CNN_8_x2(nn.Module):
    def __init__(self, num_classes):
        super(CNN_8_x2, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 8, 5) # input features, output features, kernel size
        self.act1 = nn.ReLU()
        self.mp1 = nn.MaxPool2d(2, 2) # kernel size, stride
        
        self.conv2 = nn.Conv2d(8, 16, 5) # input features, output features, kernel size
        self.act2 = nn.ReLU()
        self.mp2 = nn.MaxPool2d(2, 2) # kernel size, stride
        
        self.fc = nn.Linear(16*93*93, num_classes) # 4x4 is the remaining spatial resolution here

    def forward(self, x):
        x = self.mp1(self.act1(self.conv1(x)))
        x = self.mp2(self.act2(self.conv2(x)))
        # The view flattens the output to a vector (the representation needed by the classifier)
        x = x.view(-1, 16*93*93)
        x = self.fc(x)
        return x