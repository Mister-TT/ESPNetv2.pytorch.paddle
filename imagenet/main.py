import paddle
import argparse
import Model as Net
import numpy as np
from utils import *
import random
import os
from LRSchedule import MyLRScheduler
__author__ = 'Sachin Mehta'
__license__ = 'MIT'
__maintainer__ = 'Sachin Mehta'


def compute_params(model):
    return sum([np.prod(p.shape) for p in model.parameters()])


def main(args):
    best_prec1 = 0.0
    if not os.path.isdir(args.savedir):
        os.mkdir(args.savedir)
    model = Net.EESPNet(classes=1000, s=args.s)
    print('Network Parameters: ' + str(compute_params(model)))
    cuda_available = paddle.device.cuda.device_count() >= 1
    num_gpus = paddle.device.cuda.device_count()
    if num_gpus >= 1:
        model = paddle.DataParallel(model)
    if cuda_available:
        model = model
    logFileLoc = args.savedir + 'logs.txt'
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write('\n%s\t%s\t%s\t%s\t%s\t' % ('Epoch', 'Loss(Tr)',
            'Loss(val)', 'top1 (tr)', 'top1 (val'))
    optimizer = paddle.optimizer.SGD(model.parameters(), args.lr, momentum=args.
        momentum, weight_decay=args.weight_decay, nesterov=True)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = paddle.load(path=args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.set_state_dict(state_dict=checkpoint['state_dict'])
            optimizer.set_state_dict(state_dict=checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume,
                checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = paddle.vision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    train_loader1 = paddle.io.DataLoader(
        paddle.vision.datasets.ImageFolder(traindir, paddle.vision.transforms.Compose([
                paddle.vision.transforms.RandomResizedCrop(args.inpSize), 
                paddle.vision.transforms.RandomHorizontalFlip(), 
                paddle.vision.transforms.ToTensor(),
        normalize])), batch_size=args.batch_size, shuffle=True, 
        num_workers=args.workers, pin_memory=True)


    val_loader = paddle.io.DataLoader(
        paddle.vision.datasets.ImageFolder(valdir, paddle.vision.transforms.Compose([
            paddle.vision.transforms.Resize(int(args.inpSize / 0.875)), 
            paddle.vision.transforms.CenterCrop(args.inpSize), 
            paddle.vision.transforms.ToTensor(),
        normalize])), batch_size=args.batch_size, shuffle=False, 
        num_workers=args.workers, pin_memory=True)
    
    step_sizes = [51, 101, 131, 161, 191, 221, 251, 281]
    step_store = list()
    for step in step_sizes:
        step_store.append(step - 1)
    customLR = MyLRScheduler(args.lr, 5, step_sizes)
    if args.start_epoch != 0:
        for epoch in range(args.start_epoch):
            customLR.get_lr(epoch)
    for epoch in range(args.start_epoch, args.epochs):
        lr_log = customLR.get_lr(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_log
        print('LR for epoch {} = {:.5f}'.format(epoch, lr_log))
        train_prec1, train_loss = train(train_loader1, model, optimizer, epoch)
        val_prec1, val_loss = validate(val_loader, model)
        is_best = val_prec1.item() > best_prec1
        best_prec1 = max(val_prec1.item(), best_prec1)
        back_check = True if epoch in step_store else False
        save_checkpoint({'epoch': epoch + 1, 'arch': 'ESPNet', 'state_dict':
            model.state_dict(), 'best_prec1': best_prec1, 'optimizer':
            optimizer.state_dict()}, is_best, back_check, epoch, args.savedir)
        logger.write('\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.7f' % (
            epoch, train_loss, val_loss, train_prec1, val_prec1, lr_log))
        logger.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
        'ESPNetv2 Training on the ImageNet')
    parser.add_argument('--data', default=
        '/home/ubuntu/ILSVRC2015/Data/CLS-LOC/', help='path to dataset')
    parser.add_argument('--workers', default=12, type=int, help=
        'number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=300, type=int, help=
        'number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, help=
        'manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=512, type=int, help=
        'mini-batch size (default: 512)')
    parser.add_argument('--lr', default=0.1, type=float, help=
        'initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', default=4e-05, type=float, help=
        'weight decay (default: 4e-5)')
    parser.add_argument('--resume', default='', type=str, help=
        'path to latest checkpoint (default: none)')
    parser.add_argument('--savedir', type=str, default='./results', help=
        'Location to save the results')
    parser.add_argument('--s', default=1, type=float, help=
        'Factor by which output channels should be reduced (s > 1 for increasing the dims while < 1 for decreasing)'
        )
    parser.add_argument('--inpSize', default=224, type=int, help=
        'Input image size (default: 224 x 224)')
    args = parser.parse_args()
    args.parallel = True

    random.seed(1882)
    paddle.seed(seed=1882)
    args.savedir = args.savedir + '_s_' + str(args.s) + '_inp_' + str(args.
        inpSize) + os.sep
    main(args)
