import sys
sys.path.append('/home/aistudio/workspace/ESPNetv2.pytorch.paddle/utils')
import paddle_aux
import paddle
__author__ = 'Sachin Mehta'
__license__ = 'MIT'
__maintainer__ = 'Sachin Mehta'
from IOUEval import iouEval
import time
import numpy as np


def poly_lr_scheduler(args, optimizer, epoch, power=0.9):
    lr = round(args.lr * (1 - epoch / args.max_epochs) ** power, 8)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def val(args, val_loader, model, criterion):
    """
    :param args: general arguments
    :param val_loader: loaded for validation dataset
    :param model: model
    :param criterion: loss function
    :return: average epoch loss, overall pixel-wise accuracy, per class accuracy, per class iu, and mIOU
    """
    model.eval()
    iouEvalVal = iouEval(args.classes)
    epoch_loss = []
    total_batches = len(val_loader)
    for i, (input, target) in enumerate(val_loader):
        start_time = time.time()
        if args.onGPU:
            input = input
            target = target
        output1 = model(input)
        loss = criterion(output1, target)
        epoch_loss.append(loss.item())
        time_taken = time.time() - start_time
        iouEvalVal.addBatch(output1.max(1)[1].data, target.data)
        print('[%d/%d] loss: %.3f time: %.2f' % (i, total_batches, loss.
            item(), time_taken))
    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)
    overall_acc, per_class_acc, per_class_iu, mIOU = iouEvalVal.getMetric()
    return (average_epoch_loss_val, overall_acc, per_class_acc,
        per_class_iu, mIOU)


def train(args, train_loader, model, criterion, optimizer, epoch):
    """
    :param args: general arguments
    :param train_loader: loaded for training dataset
    :param model: model
    :param criterion: loss function
    :param optimizer: optimization algo, such as ADAM or SGD
    :param epoch: epoch number
    :return: average epoch loss, overall pixel-wise accuracy, per class accuracy, per class iu, and mIOU
    """
    model.train()
    iouEvalTrain = iouEval(args.classes)
    epoch_loss = []
    total_batches = len(train_loader)
    for i, (input, target) in enumerate(train_loader):
        start_time = time.time()
        if args.onGPU:
            input = input
            target = target
        output1, output2 = model(input)
        """Class Method: *.zero_grad, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
        optimizer.clear_grad()
        loss1 = criterion(output1, target)
        loss2 = criterion(output2, target)
        loss = loss1 + loss2
        """Class Method: *.zero_grad, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
        optimizer.clear_grad()
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())
        time_taken = time.time() - start_time
        iouEvalTrain.addBatch(output1.max(1)[1].data, target.data)
        print('[%d/%d] loss: %.3f time:%.2f' % (i, total_batches, loss.item
            (), time_taken))
    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
    overall_acc, per_class_acc, per_class_iu, mIOU = iouEvalTrain.getMetric()
    return (average_epoch_loss_train, overall_acc, per_class_acc,
        per_class_iu, mIOU)


def save_checkpoint(state, filenameCheckpoint='checkpoint.pth.tar'):
    """
    helper function to save the checkpoint
    :param state: model state
    :param filenameCheckpoint: where to save the checkpoint
    :return: nothing
    """
    paddle.save(obj=state, path=filenameCheckpoint)


def netParams(model):
    """
    helper function to see total network parameters
    :param model: model
    :return: total network parameters
    """
    return np.sum([np.prod(parameter.shape) for parameter in model.
        parameters()])
