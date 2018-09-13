
# coding: utf-8

from torchnet import meter

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
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    for i, data in enumerate(train_loader):
        inputs, masks, targets = data
        inputs = inputs.to(device).float()
        targets = targets.to(device).long()
        

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        #prec1 = accuracy(outputs, targets)
        losses.update(loss.item(), inputs.size(0))
        #top1.update(prec1, inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 200 == 0:
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

    confusion = meter.ConfusionMeter(len(val_loader.dataset.class_to_idx), normalized=True)

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

    writer.add_scalars('confusion matrix', {'00' : confusion.value()[0,0], 
                                            '01' : confusion.value()[0,1],
                                            '10' : confusion.value()[1,0],
                                            '11' : confusion.value()[1,1]}, epoch)
    writer.add_scalar('Accuracy', top1.avg, epoch)
    print(' * Validation accuracy: Prec@1 {top1.avg:.3f} '.format(top1=top1))
    print('Confusion matrix: ', confusion.value())
