import paddle
import argparse
import time
import Model as Net
import numpy as np
from utils import *
import os
False = True
"""
This file is mostly adapted from the PyTorch ImageNet example
"""
__author__ = 'Sachin Mehta'
__license__ = 'MIT'
__maintainer__ = 'Sachin Mehta'


def main(args):
    model = Net.EESPNet(classes=1000, s=args.s)
    model = paddle.DataParallel(model)
    if not os.path.isfile(args.weightFile):
        print('Weight file does not exist')
        exit(-1)
    dict_model = paddle.load(path=args.weightFile)
    model.set_state_dict(state_dict=dict_model)
    n_params = sum([np.prod(p.shape) for p in model.parameters()])
    print('Parameters: ' + str(n_params))
    valdir = os.path.join(args.data, 'val')
    traindir = os.path.join(args.data, 'train')
    normalize = paddle.vision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    val_loader = paddle.io.DataLoader(
        paddle.vision.datasets.ImageFolder(valdir, paddle.vision.transforms.Compose([
            paddle.vision.transforms.Resize(int(args.inpSize / 0.875)), 
            paddle.vision.transforms.CenterCrop(args.inpSize), 
            paddle.vision.transforms.ToTensor(), normalize
        ])), batch_size=args.batch_size, shuffle=False, 
        num_workers=args.workers, pin_memory=True)
    validate(val_loader, model)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
        'ESPNetv2 Training on the ImageNet')
    parser.add_argument('--data', default=
        '/home/ubuntu/ILSVRC2015/Data/CLS-LOC/', help='path to dataset')
    parser.add_argument('--workers', default=12, type=int, help=
        'number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=512, type=int,
        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--print-freq', '-p', default=10, type=int, metavar
        ='N', help='print frequency (default: 10)')
    parser.add_argument('--s', default=1, type=float, help=
        'Factor by which output channels should be reduced (s > 1 for increasing the dims while < 1 for decreasing)'
        )
    parser.add_argument('--weightFile', type=str, default='', help=
        'weight file')
    parser.add_argument('--inpSize', default=224, type=int, help='Input size')
    args = parser.parse_args()
    args.parallel = True
    main(args)
