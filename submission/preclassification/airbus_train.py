'''
Preclassification: train the model.
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

# Define model, optimizer and loss function (criterion)
model = CNN_8_x4(2).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001,
                            momentum=0.9,
                            weight_decay=5e-4)

writer = SummaryWriter('runs/CNN_8_x4_nb')

# Train the model
total_epochs = 50
for epoch in range(total_epochs):
    print("EPOCH:", epoch + 1, time.strftime("%Y-%m-%d %H:%M:%S"))
    print("TRAIN")
    train(dataset.train_loader, model, criterion, optimizer, device, writer)
    print("VALIDATION", time.strftime("%Y-%m-%d %H:%M:%S"))
    validate(dataset.val_loader, model, criterion, device, writer, epoch)
    #torch.save(model, 'CNN_8_x4_nb.model')  # Commented to protect the trained model
writer.close()