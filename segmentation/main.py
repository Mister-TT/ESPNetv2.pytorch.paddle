import paddle
import loadData as ld
import os
import pickle
from cnn import SegmentationModel as net
import Transforms as myTransforms
import DataSet as myDataLoader
from argparse import ArgumentParser
from train_utils import train, val, netParams, save_checkpoint, poly_lr_scheduler
__author__ = 'Sachin Mehta'
__license__ = 'MIT'
__maintainer__ = 'Sachin Mehta'


def trainValidateSegmentation(args):
    """
    Main function for trainign and validation
    :param args: global arguments
    :return: None
    """
    cuda_available = paddle.device.cuda.device_count() >= 1
    num_gpus = paddle.device.cuda.device_count()
    model = net.EESPNet_Seg(args.classes, s=args.s, pretrained=args.
        pretrained, gpus=num_gpus)
    if num_gpus >= 1:
        model = paddle.DataParallel(model)
    args.savedir = args.savedir + str(args.s) + '/'
    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)
    if not os.path.isfile(args.cached_data_file):
        dataLoad = ld.LoadData(args.data_dir, args.classes, args.
            cached_data_file)
        data = dataLoad.processData()
        if data is None:
            print('Error while pickling data. Please check.')
            exit(-1)
    else:
        data = pickle.load(open(args.cached_data_file, 'rb'))
    if cuda_available:
        args.onGPU = True
        model = model
    total_paramters = netParams(model)
    print('Total network parameters: ' + str(total_paramters))
    weight = paddle.to_tensor(data=data['classWeights'])
    if args.onGPU:
        weight = weight
    criteria = paddle.nn.CrossEntropyLoss(weight=weight)
    if args.onGPU:
        criteria = criteria
    print('Data statistics')
    print(data['mean'], data['std'])
    print(data['classWeights'])
    trainDataset_main = myTransforms.Compose([myTransforms.Normalize(mean=
        data['mean'], std=data['std']), myTransforms.RandomCropResize(size=
        (args.inWidth, args.inHeight)), myTransforms.RandomFlip(),
        myTransforms.ToTensor(args.scaleIn)])
    trainDataset_scale1 = myTransforms.Compose([myTransforms.Normalize(mean
        =data['mean'], std=data['std']), myTransforms.RandomCropResize(size
        =(int(args.inWidth * 1.5), int(1.5 * args.inHeight))), myTransforms
        .RandomFlip(), myTransforms.ToTensor(args.scaleIn)])
    trainDataset_scale2 = myTransforms.Compose([myTransforms.Normalize(mean
        =data['mean'], std=data['std']), myTransforms.RandomCropResize(size
        =(int(args.inWidth * 1.25), int(1.25 * args.inHeight))),
        myTransforms.RandomFlip(), myTransforms.ToTensor(args.scaleIn)])
    trainDataset_scale3 = myTransforms.Compose([myTransforms.Normalize(mean
        =data['mean'], std=data['std']), myTransforms.RandomCropResize(size
        =(int(args.inWidth * 0.75), int(0.75 * args.inHeight))),
        myTransforms.RandomFlip(), myTransforms.ToTensor(args.scaleIn)])
    trainDataset_scale4 = myTransforms.Compose([myTransforms.Normalize(mean
        =data['mean'], std=data['std']), myTransforms.RandomCropResize(size
        =(int(args.inWidth * 0.5), int(0.5 * args.inHeight))), myTransforms
        .RandomFlip(), myTransforms.ToTensor(args.scaleIn)])
    valDataset = myTransforms.Compose([myTransforms.Normalize(mean=data[
        'mean'], std=data['std']), myTransforms.Scale(1024, 512),
        myTransforms.ToTensor(args.scaleIn)])
>>>>>>    trainLoader = torch.utils.data.DataLoader(myDataLoader.MyDataset(data[
        'trainIm'], data['trainAnnot'], transform=trainDataset_main),
        batch_size=args.batch_size, shuffle=True, num_workers=args.
        num_workers, pin_memory=True)
>>>>>>    trainLoader_scale1 = torch.utils.data.DataLoader(myDataLoader.MyDataset
        (data['trainIm'], data['trainAnnot'], transform=trainDataset_scale1
        ), batch_size=args.batch_size, shuffle=True, num_workers=args.
        num_workers, pin_memory=True)
>>>>>>    trainLoader_scale2 = torch.utils.data.DataLoader(myDataLoader.MyDataset
        (data['trainIm'], data['trainAnnot'], transform=trainDataset_scale2
        ), batch_size=args.batch_size, shuffle=True, num_workers=args.
        num_workers, pin_memory=True)
>>>>>>    trainLoader_scale3 = torch.utils.data.DataLoader(myDataLoader.MyDataset
        (data['trainIm'], data['trainAnnot'], transform=trainDataset_scale3
        ), batch_size=args.batch_size, shuffle=True, num_workers=args.
        num_workers, pin_memory=True)
>>>>>>    trainLoader_scale4 = torch.utils.data.DataLoader(myDataLoader.MyDataset
        (data['trainIm'], data['trainAnnot'], transform=trainDataset_scale4
        ), batch_size=args.batch_size, shuffle=True, num_workers=args.
        num_workers, pin_memory=True)
>>>>>>    valLoader = torch.utils.data.DataLoader(myDataLoader.MyDataset(data[
        'valIm'], data['valAnnot'], transform=valDataset), batch_size=args.
        batch_size, shuffle=False, num_workers=args.num_workers, pin_memory
        =True)
    if args.onGPU:
        False = True
    start_epoch = 0
    best_val = 0
    lr = args.lr
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(),
        learning_rate=lr, epsilon=1e-08, weight_decay=0.0005, beta1=(0.9, 
        0.999)[0], beta2=(0.9, 0.999)[1])
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = paddle.load(path=args.resume)
            start_epoch = checkpoint['epoch']
            best_val = checkpoint['best_val']
            model.set_state_dict(state_dict=checkpoint['state_dict'])
            optimizer.set_state_dict(state_dict=checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume,
                checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    logFileLoc = args.savedir + args.logFile
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write('Parameters: %s' % str(total_paramters))
        logger.write('\n%s\t%s\t%s\t%s\t%s\t' % ('Epoch', 'Loss(Tr)',
            'Loss(val)', 'mIOU (tr)', 'mIOU (val'))
    logger.flush()
    for epoch in range(start_epoch, args.max_epochs):
        poly_lr_scheduler(args, optimizer, epoch)
        lr = 0
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        print('Learning rate: ' + str(lr))
        train(args, trainLoader_scale1, model, criteria, optimizer, epoch)
        train(args, trainLoader_scale2, model, criteria, optimizer, epoch)
        train(args, trainLoader_scale4, model, criteria, optimizer, epoch)
        train(args, trainLoader_scale3, model, criteria, optimizer, epoch)
        (lossTr, overall_acc_tr, per_class_acc_tr, per_class_iu_tr, mIOU_tr
            ) = train(args, trainLoader, model, criteria, optimizer, epoch)
        (lossVal, overall_acc_val, per_class_acc_val, per_class_iu_val,
            mIOU_val) = val(args, valLoader, model, criteria)
        is_best = mIOU_val > best_val
        best_val = max(mIOU_val, best_val)
        save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict
            (), 'optimizer': optimizer.state_dict(), 'lr': lr, 'best_val':
            best_val}, args.savedir + 'checkpoint.pth.tar')
        if is_best:
            model_file_name = args.savedir + os.sep + 'model_best.pth'
            paddle.save(obj=model.state_dict(), path=model_file_name)
        with open(args.savedir + 'acc_' + str(epoch) + '.txt', 'w') as log:
            log.write(
                """
Epoch: %d	 Overall Acc (Tr): %.4f	 Overall Acc (Val): %.4f	 mIOU (Tr): %.4f	 mIOU (Val): %.4f"""
                 % (epoch, overall_acc_tr, overall_acc_val, mIOU_tr, mIOU_val))
            log.write('\n')
            log.write('Per Class Training Acc: ' + str(per_class_acc_tr))
            log.write('\n')
            log.write('Per Class Validation Acc: ' + str(per_class_acc_val))
            log.write('\n')
            log.write('Per Class Training mIOU: ' + str(per_class_iu_tr))
            log.write('\n')
            log.write('Per Class Validation mIOU: ' + str(per_class_iu_val))
        logger.write('\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.7f' % (
            epoch, lossTr, lossVal, mIOU_tr, mIOU_val, lr))
        logger.flush()
        print('Epoch : ' + str(epoch) + ' Details')
        print(
            """
Epoch No.: %d	Train Loss = %.4f	Val Loss = %.4f	 mIOU(tr) = %.4f	 mIOU(val) = %.4f"""
             % (epoch, lossTr, lossVal, mIOU_tr, mIOU_val))
    logger.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', default='ESPNetv2', help='Model name')
    parser.add_argument('--data_dir', default='./city', help='Data directory')
    parser.add_argument('--inWidth', type=int, default=1024, help=
        'Width of RGB image')
    parser.add_argument('--inHeight', type=int, default=512, help=
        'Height of RGB image')
    parser.add_argument('--scaleIn', type=int, default=1, help=
        'For ESPNet-C, scaleIn=8. For ESPNet, scaleIn=1')
    parser.add_argument('--max_epochs', type=int, default=300, help=
        'Max. number of epochs')
    parser.add_argument('--num_workers', type=int, default=12, help=
        'No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=10, help=
        'Batch size. 12 for ESPNet-C and 6 for ESPNet. Change as per the GPU memory'
        )
    parser.add_argument('--step_loss', type=int, default=100, help=
        'Decrease learning rate after how many epochs.')
    parser.add_argument('--lr', type=float, default=0.0005, help=
        'Initial learning rate')
    parser.add_argument('--savedir', default='./results_espnetv2_', help=
        'directory to save the results')
    parser.add_argument('--resume', type=str, default='', help=
        'Use this flag to load last checkpoint for training')
    parser.add_argument('--classes', type=int, default=20, help=
        'No of classes in the dataset. 20 for cityscapes')
    parser.add_argument('--cached_data_file', default='city.p', help=
        'Cached file name')
    parser.add_argument('--logFile', default='trainValLog.txt', help=
        'File that stores the training and validation logs')
    parser.add_argument('--pretrained', default='', help=
        'Pretrained ESPNetv2 weights.')
    parser.add_argument('--s', default=1, type=float, help='scaling parameter')
    trainValidateSegmentation(parser.parse_args())
