'''
Preclassification: compare different softmax thresholds.
'''
# coding: utf-8

import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorboardX import SummaryWriter

from airbus_dataloader import *
from airbus_train_val_functions import *
from airbus_models import *

# Define the arguments
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_gpu = torch.cuda.is_available()
batch_size = 4
workers = 4
path = '../airbus/'
aug=True
resize_factor=4
empty_frac=1
test_size=0.1

# Create data sets/loaders
dataset = AirbusDS(torch.cuda.is_available(), batch_size, workers, path, aug, resize_factor, empty_frac, test_size)

# Load a model
model = torch.load('CNN_8_x4_bal.model').to(device)

# Define optimizer and loss function (criterion)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001,
                            momentum=0.9,
                            weight_decay=5e-4)

# tensorboard's writer
writer = SummaryWriter('runs/thresholds/CNN_8_x4_bal')

# Run validation with different thresholds
for t in np.arange(0.1, 1.0, 0.1):    
    print("VALIDATION: threshold=", t, time.strftime("%Y-%m-%d %H:%M:%S"))
    validate_soft(dataset.val_loader, model, criterion, device, writer, epoch=0, threshold=t)
    
writer.close()