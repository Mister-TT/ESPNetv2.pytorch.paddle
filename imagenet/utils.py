import paddle
import os
import shutil
import time
__author__ = 'Sachin Mehta'
__license__ = 'MIT'
__maintainer__ = 'Sachin Mehta'
"""
This file is mostly adapted from the PyTorch ImageNet example
"""


class AverageMeter(object):
    """Computes and stores the average and current value"""

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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.shape[0]
    _, pred = output.topk(k=maxk, axis=1, largest=True, sorted=True)
    pred = pred.t()
    """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
    correct = pred.equal(y=target.view(1, -1).expand_as(y=pred))
    res = []
    for k in topk:
        """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
        correct_k = correct[:k].view(-1).astype(dtype='float32').sum(axis=0,
            keepdim=True)
        res.append(correct_k.multiply_(y=paddle.to_tensor(100.0 / batch_size)))
    return res


"""
Utility to save checkpoint or not
"""


def save_checkpoint(state, is_best, back_check, epoch, dir):
    check_pt_file = dir + os.sep + 'checkpoint.pth.tar'
    paddle.save(obj=state, path=check_pt_file)
    if is_best:
        paddle.save(obj=state['state_dict'], path=dir + os.sep +
            'model_best.pth')
    if back_check:
        shutil.copyfile(check_pt_file, dir + os.sep + 'checkpoint_back' +
            str(epoch) + '.pth.tar')


"""
Cross entropy loss function
"""


def loss_fn(outputs, labels):
    return paddle.nn.functional.cross_entropy(input=outputs, label=labels)


"""
Training loop
"""


def train(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        input = input
        target = target
        output = model(input)
        try:
            input_spec = list(paddle.static.InputSpec.from_tensor(paddle.to_tensor(t)) for t in (input, ))
            paddle.jit.save(model, input_spec=input_spec, path="./model")
            print('[JIT] torch.export successed.')
            exit(0)
        except Exception as e:
            print('[JIT] torch.export failed.')
            raise e
        loss = loss_fn(output, target)
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.shape[0])
        top1.update(prec1[0], input.shape[0])
        top5.update(prec5[0], input.shape[0])
        """Class Method: *.zero_grad, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
        optimizer.clear_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        if i % 100 == 0:
            print(
                'Epoch: %d[%d/%d]\t\tBatch Time:%.4f\t\tLoss:%.4f\t\ttop1:%.4f (%.4f)\t\ttop5:%.4f (%.4f)'
                 % (epoch, i, len(train_loader), batch_time.avg, losses.avg,
                top1.val, top1.avg, top5.val, top5.avg))
    return top1.avg, losses.avg


"""
Validation loop
"""


def validate(val_loader, model):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    end = time.time()
    with paddle.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input
            target = target
            output = model(input)
            loss = loss_fn(output, target)
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.shape[0])
            top1.update(prec1[0], input.shape[0])
            top5.update(prec5[0], input.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if i % 100 == 0:
                print(
                    'Batch:[%d/%d]\t\tBatchTime:%.3f\t\tLoss:%.3f\t\ttop1:%.3f (%.3f)\t\ttop5:%.3f(%.3f)'
                     % (i, len(val_loader), batch_time.avg, losses.avg,
                    top1.val, top1.avg, top5.val, top5.avg))
        print(' * Prec@1:%.3f Prec@5:%.3f' % (top1.avg, top5.avg))
        return top1.avg, losses.avg
