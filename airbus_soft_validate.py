# coding: utf-8

import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorboardX import SummaryWriter

from airbus_dataloader import *
from airbus_train_val_functions import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_gpu = torch.cuda.is_available()
batch_size = 4
workers = 4
path = '../airbus/'
aug=True
resize_factor=4
empty_frac=1
test_size=0.1
    
dataset = AirbusDS(torch.cuda.is_available(), batch_size, workers, path, aug, resize_factor, empty_frac, test_size)

# Define optimizer and loss function (criterion)
#model = CNN_8_x4(2).to(device)
model = torch.load('CNN_8_x4_notbalanced.model').to(device)
criterion = nn.CrossEntropyLoss().to(device)

# we can use advanced stochastic gradient descent algorithms 
# with regularization (weight-decay) or momentum
optimizer = torch.optim.SGD(model.parameters(), lr=0.001,
                            momentum=0.9,
                            weight_decay=5e-4)

writer = SummaryWriter('runs/tresholds')

for t in np.arange(0.1, 1.0, 0.1):
    
    print("VALIDATION: treshold=", t, time.strftime("%Y-%m-%d %H:%M:%S"))
    validate_soft(dataset.val_loader, model, criterion, device, writer, epoch=0, treshold=t)
    
writer.close()