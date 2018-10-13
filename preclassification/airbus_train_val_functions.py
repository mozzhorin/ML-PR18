'''
Preclassification: functions for training and validating.
'''
# coding: utf-8

from torchnet import meter

import torch
import numpy as np
import pandas as pd
import time
import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_gpu = torch.cuda.is_available()

class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(outputs, targets):
    """
    Evaluates a model's top k accuracy

    Parameters:
        outputs (torch.autograd.Variable): model output
        targets (torch.autograd.Variable): ground-truths/labels
        

    Returns:
        float: percentage of correct predictions
    """

    batch_size = targets.size(0)

    _, pred = torch.max(outputs.data, 1)
    correct = (pred == targets).sum().item()

    res = 100 * correct / batch_size
    return res



def train(train_loader, model, criterion, optimizer, device, writer):
    """
    Trains/updates the model for one epoch on the training dataset.

    Parameters:
        train_loader (torch.utils.data.DataLoader): The trainset dataloader
        model (torch.nn.module): Model to be trained
        criterion (torch.nn.criterion): Loss function
        optimizer (torch.optim.optimizer): optimizer instance like SGD or Adam
        device (string): cuda or cpu
    """

    losses = AverageMeter()

    # switch to train mode
    model.train()

    for i, data in enumerate(train_loader):
        inputs, masks, targets = data
        inputs = inputs.to(device).float()
        targets = targets.to(device).long()
        

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure the loss        
        losses.update(loss.item(), inputs.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 200 == 199:
            writer.add_scalar('loss', losses.avg, i)
            print('Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(loss=losses))


def validate(val_loader, model, criterion, device, writer, epoch=0):
    """
    Evaluates/validates the model

    Parameters:
        val_loader (torch.utils.data.DataLoader): The validation or testset dataloader
        model (torch.nn.module): Model to be evaluated/validated
        criterion (torch.nn.criterion): Loss function
        device (string): cuda or cpu
    """

    losses = AverageMeter()
    top1 = AverageMeter()

    confusion = meter.ConfusionMeter(len(val_loader.dataset.class_to_idx), normalized=False)

    # switch to evaluate mode
    model.eval()

    for i, data in enumerate(val_loader):
        inputs, targets = data
        inputs = inputs.to(device).float()
        targets = targets.to(device).long()

        # compute output
        outputs = model(inputs)

        # compute loss
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1 = accuracy(outputs, targets)
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1, inputs.size(0))

        # add to confusion matrix
        confusion.add(outputs.data, targets)

    count = losses.count
    writer.add_scalars('confusion matrix', {'00' : confusion.value()[0,0]/count, 
                                            '01' : confusion.value()[0,1]/count,
                                            '10' : confusion.value()[1,0]/count,
                                            '11' : confusion.value()[1,1]/count}, epoch)
    writer.add_scalar('Accuracy', top1.avg, epoch)
    print(' * Validation accuracy: Prec@1 {top1.avg:.3f} '.format(top1=top1))
    print('Confusion matrix: ', confusion.value()/count)
    

def soft_class(outputs, threshold):
    '''Choose the class of image from 0 and 1 using softmax function
    and a given threshold.'''
    
    softout = torch.nn.functional.softmax(outputs, dim=1)
    result = []
    for t in softout:
        if t[1]>threshold :
            result.append(1)
        else:
            result.append(0)
    return(torch.Tensor(result).to(device).long())

def accuracy_soft(outputs, targets):
    '''Calculate model accuracy using a threshold on softmax function.'''

    batch_size = targets.size(0)
    correct = (outputs == targets).sum().item()

    res = 100 * correct / batch_size
    return res

def validate_soft(val_loader, model, criterion, device, writer, epoch=0, threshold=0.5):
    """
    Evaluates/validates the model using a threshold on softmax function.

    Parameters:
        val_loader (torch.utils.data.DataLoader): The validation or testset dataloader
        model (torch.nn.module): Model to be evaluated/validated
        criterion (torch.nn.criterion): Loss function
        device (string): cuda or cpu
        writer (tensorboardX.SummaryWriter): a tensorboard SummaryWriter
        epoch (int): the epoch counter
        threshold (float): the threshold on softmax function.
        
    """

    losses = AverageMeter()
    top1 = AverageMeter()

    confusion = meter.ConfusionMeter(len(val_loader.dataset.class_to_idx), normalized=False)

    # switch to evaluate mode
    model.eval()

    for i, data in enumerate(val_loader):
        inputs, targets = data
        inputs = inputs.to(device).float()
        targets = targets.to(device).long()

        # compute output
        outputs = model(inputs)

        # compute loss
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        outputs = soft_class(outputs, threshold)
        prec1 = accuracy_soft(outputs, targets)
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1, inputs.size(0))

        # add to confusion matrix
        confusion.add(outputs.data, targets)

    count = losses.count
    writer.add_scalars('confusion matrix', {'00' : confusion.value()[0,0]/count, 
                                            '01' : confusion.value()[0,1]/count,
                                            '10' : confusion.value()[1,0]/count,
                                            '11' : confusion.value()[1,1]/count}, epoch)
    writer.add_scalar('Accuracy', top1.avg, epoch)
    print(' * Validation accuracy: Prec@1 {top1.avg:.3f} '.format(top1=top1))
    print('Confusion matrix: ', confusion.value()/count)
    
def predict(test_loader, model, device, predict_file, threshold=0.5,):
    """
    Evaluates/validates the model

    Parameters:
        test_loader (torch.utils.data.DataLoader): The testset dataloader
        model (torch.nn.module): Model to be evaluated/validated
        criterion (torch.nn.criterion): Loss function
        device (string): cuda or cpu
        predict_file (str): file name for saving the whole prediction (image names and classes).
        threshold (float): the threshold on softmax function.
    """
    
    logging.basicConfig(filename='predict.log', level=logging.INFO,
                        format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S ')
    logger = logging.getLogger("pred_log")
    
    # switch to evaluate mode
    model.eval()
    
    test_size = len(test_loader)
    
    predictions = pd.DataFrame(columns=['ImageId', 'Label'])
    print('START PREDICTIONS', time.strftime("%Y-%m-%d %H:%M:%S"))
    logger.info('START PREDICTIONS')
    start_time = time.time()

    for i, data in enumerate(test_loader):
        inputs, img_id = data
        inputs = inputs.to(device).float()

        # compute output
        outputs = model(inputs)
        outputs = soft_class(outputs, threshold)
        predictions = predictions.append(pd.DataFrame(\
                      np.array([img_id, outputs.cpu().numpy()]).T,columns=['ImageId', 'Label']))
        
        if i % 1000 == 0:
            print('Predicted {num} from {size}\t'.format(num=i, size=test_size))
            logger.info('Predicted %s from %s images.', i, test_size)
            
    predictions.to_csv(predict_file, sep='\t', header=False, index=False)
    predictions[predictions['Label']!='0']['ImageId'].to_csv('filtered_ships_v2.txt', sep='\t', header=False, index=False)
    run_time = time.time() - start_time
    print('FINISH PREDICTIONS', time.strftime("%Y-%m-%d %H:%M:%S"))
    logger.info('FINISH PREDICTIONS')
    print('PREDICTIONS RUN TIME (s)', run_time)
    logger.info('PREDICTIONS RUN TIME %s (s)', run_time)
    #return predictions
