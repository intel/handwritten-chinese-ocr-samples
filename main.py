"""
3-Clause BSD license
Copyright (C) <2018-2021> Intel Corporation
SPDX-License-Identifier: BSD-3-Clause

From PyTorch:
Copyright (C) <2017-present> Facebook, Inc (Soumith Chintala)
All rights reserved.
"""


''' Built upon https://github.com/pytorch/examples/blob/master/imagenet/main.py with modification. '''
import argparse
import os
import random
import shutil
import time
import warnings
import sys
import math
import editdistance

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex import amp
except ImportError:
    raise ImportError(
        'Please install apex from https://www.github.com/nvidia/apex.'
    )

try:
    from warpctc_pytorch import CTCLoss
except ImportError:
    raise ImportError(
        'Please install warpctc from https://github.com/SeanNaren/warp-ctc.'
    )

from utils.dataset import ImageDataset, AlignCollate
from models.handwritten_ctr_model import hctr_model
from utils.ctc_codec import ctc_codec


def build_argparser():
    parser = argparse.ArgumentParser(description='PyTorch OCR textline Training')
    args = parser.add_argument_group('Options')
    args.add_argument('-m', '--model-type', type=str, required=True,
                      choices=['hctr'],
                      help='target model for different languages and scenarios')
    args.add_argument('-d', '--data', metavar='DIR',
                      help='path to dataset')
    args.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                      help='number of data loading workers')
    args.add_argument('-b', '--batch-size', default=8, type=int, metavar='N',
                      help='mini-batch size')
    args.add_argument('-lr', '--learning-rate', default=0.001, type=float, metavar='LR',
                      help='initial learning rate', dest='lr')
    args.add_argument('-mm', '--momentum', default=0.9, type=float, metavar='M',
                      help='momentum')
    args.add_argument('-wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                      help='weight decay')
    args.add_argument('-pf', '--print-freq', default=1000, type=int, metavar='N',
                      help='print frequency')
    args.add_argument('-vf', '--val-freq', default=50000, type=int, metavar='N',
                      help='validate frequency')
    args.add_argument('-re', '--resume', default='', type=str, metavar='PATH',
                      help='path to latest checkpoint')
    args.add_argument('-te', '--test', action='store_true',
                      help='test model on test set')
    args.add_argument('-tv', '--testverbose', action='store_true',
                      help='output result when testing')
    args.add_argument('-ep', '--epochs', default=90, type=int, metavar='N',
                      help='number of total epochs to run')
    args.add_argument('--start-epoch', default=0, type=int, metavar='N',
                      help='manual epoch number')
    args.add_argument('--seed', default=None, type=int,
                      help='seed for initializing training')
    args.add_argument('--gpu', default=None, type=int,
                      help='GPU id to use')
    args.add_argument('--world-size', default=1, type=int,
                      help='number of nodes for distributed training')
    args.add_argument('--rank', default=0, type=int,
                      help='node rank for distributed training')
    args.add_argument('--dist-url', default='env://', type=str,
                      help='url used to set up distributed training')
    args.add_argument('--dist-backend', default='nccl', type=str,
                      help='distributed backend')
    args.add_argument('--multiprocessing-distributed', action='store_true',
                      help='Use multi-processing distributed training to launch '
                           'N processes per node, which has N GPUs. This is the '
                           'fastest way to use PyTorch for either single node or '
                           'multi node data parallel training')
    return parser


best_acc = 0
codec = None

def main():
    args = build_argparser().parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    ngpus_per_node = torch.cuda.device_count()
    if args.gpu is not None:
        args.multiprocessing_distributed = False
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable multiprocessing distributed training.')
    elif ngpus_per_node <= 1:
        raise EnvironmentError(
            'No enough GPUs for multiprocessing distributed training.'
        )
    else:
        args.multiprocessing_distributed = True

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc
    global codec

    # NOTE: Only support Single-Node with Multi-GPUs
    if args.multiprocessing_distributed:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
    args.gpu = gpu

    #######################################################################
    # create model specific info
    model, characters = get_model_info(args)
    args.img_height   = model.img_height
    args.pred         = model.pred
    args.optimizer    = model.optimizer
    args.PAD          = model.PAD
    print(model)

    # criterion
    if args.pred == 'CTC':
        codec = ctc_codec(characters)
        criterion = CTCLoss().cuda(args.gpu)
    else:
        raise ValueError('not expected prediction.')

    # optimizer
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                     weight_decay=args.weight_decay)
    else:
        raise ValueError('not expected optimizer.')

    #######################################################################
    # Initialize distributed training
    if args.multiprocessing_distributed:
        # For multiprocessing distributed training, rank needs to be the
        # global rank among all the processes
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size,
                                rank=args.rank)
        print('GPU: {} initialized done.'.format(args.gpu))
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        # Initialize Amp.
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level='O2',
                                          keep_batchnorm_fp32=True,
                                          loss_scale=1.0)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model = DDP(model, delay_allreduce=True) # apex
    else: # Single-GPU
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # Initialize Amp.
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level='O2',
                                          keep_batchnorm_fp32=True,
                                          loss_scale=1.0)

    #######################################################################
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint: {}'.format(args.resume))
            checkpoint = torch.load(
                args.resume, map_location='cuda:' + str(args.gpu)
            )
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            if args.multiprocessing_distributed:
                model.module.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('=> loaded checkpoint: {} (epoch {})'
                  .format(args.resume, checkpoint['epoch']))
        else:
            raise FileNotFoundError(
                'Valid checkpoint for resume is not found.'
            )

    #######################################################################
    # Data loading code
    AlignCollate_train = AlignCollate(imgH=args.img_height, PAD=args.PAD)
    train_dataset = ImageDataset(data_path=args.data,
                                 img_shape=(1, args.img_height),
                                 phase='train',
                                 batch_size=args.batch_size)
    if args.multiprocessing_distributed:
        train_sampler = \
            torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=args.workers,
                                               collate_fn=AlignCollate_train,
                                               pin_memory=True,
                                               sampler=train_sampler)

    AlignCollate_val = AlignCollate(imgH=args.img_height, PAD=args.PAD)
    val_dataset = ImageDataset(data_path=args.data,
                               img_shape=(1, args.img_height),
                               phase='val',
                               batch_size=args.batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             collate_fn=AlignCollate_val,
                                             pin_memory=True)

    AlignCollate_test = AlignCollate(imgH=args.img_height, PAD=args.PAD)
    test_dataset = ImageDataset(data_path=args.data,
                                img_shape=(1, args.img_height),
                                phase='test',
                                batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers,
                                              collate_fn=AlignCollate_test,
                                              pin_memory=True)

    #######################################################################
    # test
    if args.test:
        test(test_loader, model, args)
        return

    #######################################################################
    # train
    val_acc = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.multiprocessing_distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        val_acc = train(train_loader, val_loader, model,
                        criterion, optimizer,
                        epoch, args, val_acc)

        # evaluate on test set
        acc = test(test_loader, model, args)

        # remember best acc and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.module.state_dict() if
                          args.multiprocessing_distributed else
                          model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, args, is_best, is_val=False)


def train(train_loader, val_loader, model, criterion, optimizer,
          epoch, args, val_acc):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    val_best_acc = val_acc

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(args.gpu, non_blocking=True)
        target_indexs, target_length = codec.encode(target)
        preds = model(input) # preds: WBD
        preds_sizes = torch.IntTensor([preds.size(0)] * args.batch_size)
        loss = criterion(preds,
                         torch.from_numpy(target_indexs),
                         preds_sizes,
                         torch.from_numpy(target_length))

        # TODO: how about inf loss ?
        if math.isnan(loss.item()):
            raise ValueError('Stop at NaN loss.')
        losses.update(loss.item(), input.size(0))

        # compute gradient and do optimization step
        optimizer.zero_grad()
        #loss.backward()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)

        if i % args.print_freq == 0:
            result = codec.decode(preds.cpu().detach().numpy())
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses))
            print('TRU {}'.format(target[0]))
            print('PRE {}'.format(result[0]))

        # validate during epoch
        if (i > 0) and (i % args.val_freq == 0):
            val_acc = test(val_loader, model, args)
            is_best = val_acc > val_best_acc
            val_best_acc = max(val_acc, val_best_acc)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict() if
                              args.multiprocessing_distributed else
                              model.state_dict(),
                'best_acc': val_best_acc,
                'optimizer' : optimizer.state_dict(),
            }, args, is_best, is_val=True)

            # switch to train mode
            model.train()

        # reset time for next iteration
        end = time.time()

    return val_best_acc


def test(data_loader, model, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    err_rate = AverageMeter()
    nchars = 0
    total = 0

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(data_loader): # test/val_loader
            # measure data loading time
            data_time.update(time.time() - end)

            input = input.cuda(args.gpu, non_blocking=True)
            preds = model(input)
            result = codec.decode(preds.cpu().detach().numpy())

            for j, (pre, tru) in enumerate(zip(result, target)):
                if args.testverbose:
                    print('TEST [{0}/{1}]'.format(j, i))
                    print('TEST PRE {}'.format(pre))
                    print('TEST TRU {}'.format(tru))
                if not isinstance(pre, str):
                    raise AssertionError(pre)
                if not isinstance(tru, str):
                    raise AssertionError(tru)
                errs = editdistance.eval(pre, tru)
                total += errs
                nchars += len(tru)

            if nchars == 0:
                raise ValueError(
                    'Number of label characters should not be 0.'
                )

            # compute character error rate
            CER = total * 1.0 / nchars
            err_rate.update(CER, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)

            if i % args.print_freq == 0:
                print('TEST: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Err {err_rate.val:.4f} ({err_rate.avg:.4f})\t'
                      .format(
                          i, len(data_loader), batch_time=batch_time,
                          data_time=data_time, err_rate=err_rate
                      )
                )

            # reset time for next iteration
            end = time.time()

    print('Total Test CER: {}'.format(CER))
    return 1.0 - CER


def save_checkpoint(state, args, is_best, is_val=False,
                    suffix_name='checkpoint.pth.tar'):
    if not args.multiprocessing_distributed or \
        (args.multiprocessing_distributed and args.rank == 0):
        if is_val:
            suffix_name = 'val_' + suffix_name
        current_ckp_name = args.model_type + '_' + suffix_name
        torch.save(state, current_ckp_name)
        if is_best:
            epoch_str = '_{:02d}ep_'.format(state['epoch'])
            acc_str = '{:.4f}acc_'.format(state['best_acc'])
            shutil.copyfile(current_ckp_name,
                            args.model_type +
                            epoch_str +
                            acc_str +
                            suffix_name)

    # NOTE: ignore the checkpoint from args.rank != 0
    # if args.multiprocessing_distributed.


class AverageMeter(object):
    '''Computes and stores the average and current value'''
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


def adjust_learning_rate(optimizer, epoch, args):
    '''Sets the learning rate to the initial LR decayed 
    by 10 every 30 epochs'''
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_model_info(args):
    '''Get specific model information: model, characters'''
    model = None
    characters = ''
    chars_list_file = ''
    if args.model_type == 'hctr':
        model = hctr_model()
    else:
        raise ValueError(
            'Model type: {} not supported'.format(args.model_type)
        )

    chars_list_file = os.path.join(args.data + 'chars_list.txt')
    with open(chars_list_file, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            characters += line

    return model, characters


if __name__ == '__main__':
    main()
