"""
Apache v2 license
Copyright (C) <2018-2021> Intel Corporation
SPDX-License-Identifier: Apache-2.0
"""


import argparse
import os
import sys
import time
import cv2
import editdistance

import torch

from utils.dataset import AlignCollate, ImageDataset, NormalizePAD
from utils.ctc_codec import ctc_codec
from models.handwritten_ctr_model import hctr_model
from main import AverageMeter


def build_argparser():
    parser = argparse.ArgumentParser(description='PyTorch OCR textline Testing')
    args = parser.add_argument_group('Options')
    args.add_argument('-m', '--model-type', dest='model_type',
                      choices=['hctr'], type=str, required=True,
                      help='target model for different languages and scenarios')
    args.add_argument('-f', '--model-file', dest='model_file',
                      type=str, metavar='PATH', required=True,
                      help='path to best model file')
    args.add_argument('-i', '--input', dest='input', default=None,
                      type=str, metavar='PATH',
                      help='path to input image file')
    args.add_argument('-b', '--batch-size', default=1, type=int, metavar='N',
                      help='mini-batch size')
    args.add_argument('--gpu', default=None, type=int,
                      help='GPU id to use.')
    args.add_argument('-bm', '--benchmark-mode', metavar='DIR', default=None,
                      help='provide the path of benchmark dataset')
    args.add_argument('-dm', '--decode-method', type=str, dest='method',
                      choices=['greedy-search', 'beam-search'],
                      default='beam-search')
    args.add_argument('-ss', '--skip-search', action='store_true', dest='skip_search',
                      help='whether skip high confidence characters when using beam search')
    args.add_argument('-kp', '--kenlm-path', metavar='PATH', type=str, dest='kenlm_path',
                      help='ngram model for scoring in beam search')
    args.add_argument('-utp', '--use-tfm-pred', action='store_true', dest='use_tfm_pred',
                      help='use transformer for candidates prediction')
    args.add_argument('-tp', '--transformer-path', metavar='DIR', type=str, dest='tfm_path',
                      help='transformer for candidates predicting')
    args.add_argument('-uts', '--use-tfm-score', action='store_true', dest='use_tfm_score',
                      help='use transformer for scoring in beam search')
    args.add_argument('-uov', '--use-openvino', action='store_true', dest='use_openvino',
                      help='use openvino for transformer inference in beam search')
    args.add_argument('-bs', '--beam-size', type=int, default=10, dest='beam_size',
                      help='beam size in beam search')
    args.add_argument('-sd', '--search-depth', type=int, default=10, dest='search_depth',
                      help='search depth (top-k) in beam search')
    args.add_argument('-lp', '--lm-panelty', default=1.9, type=float, dest='lm_panelty',
                      help='panelty of language model scoring in beam search')
    args.add_argument('-lb', '--len-bonus', default=5.7, type=float, dest='len_bonus',
                      help='length bonus of scoring in beam search')
    args.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                      help='number of data loading workers in benchmark mode')
    args.add_argument('-tv', '--testverbose', action='store_true',
                      help='output result when testing')
    args.add_argument('-pf', '--print-freq', default=1000, type=int, metavar='N',
                      help='print frequency')
    return parser


def test():
    args = build_argparser().parse_args()
    if os.path.isfile(args.model_file) is False:
        print(f'=> no model file found at: {args.model_file}')
        return
    if args.input == None and args.benchmark_mode == None:
        print(f'=> no input file and no benchmark dataset')
        return

    if args.input != None and len(os.listdir(args.input)) == 0 and \
       os.path.isfile(args.input) is False:
        print(f'=> no input file found at: {args.input}')
        return
    if args.gpu is not None:
        print(f'Use GPU: {args.gpu} for testing')

    # create model specific info
    model, img_height, characters = get_model_info(args.model_type)
    # depends on using ctc or attension
    codec = ctc_codec(characters)
    if args.method == 'beam-search':
        codec.set_beam_search(args.skip_search,
                              ngram_path=args.kenlm_path,
                              tfm_path=args.tfm_path,
                              lm_panelty=args.lm_panelty,
                              len_bonus=args.len_bonus,
                              beam_size=args.beam_size,
                              search_depth=args.search_depth,
                              use_tfm_score=args.use_tfm_score,
                              use_tfm_pred=args.use_tfm_pred,
                              use_openvino=args.use_openvino)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # Use CPU for testing
        model = model.cpu()

    # Load pre-trained model parameters
    print(f'=> loading model file: {args.model_file}')
    checkpoint = torch.load(args.model_file, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    # Switch to evaluate mode
    model.eval()

    if args.benchmark_mode != None:
        benchmark(model, codec, args)
        return

    # Preprocess the input images if needed
    input_list = preprocess_input(args.input, img_height)

    batch_num = len(input_list)//args.batch_size
    with torch.no_grad():
        for i in range(batch_num):
            if i % 1 == 0:
                print("batch %d is being processed..." % i)

            maxW = 0
            batch_images = input_list[i * args.batch_size: (i+1) * args.batch_size]
            padded_images = []
            for image in batch_images:
                h, w = image.shape
                if w > maxW:
                    maxW = w
            for image in batch_images:
                image = image[:, :, None]
                transform = NormalizePAD((1, img_height, maxW))
                padded_images.append(transform(image))

            batch_image_tensors = torch.cat([t.unsqueeze(0) for t in padded_images], 0)

            # Compute and decode output
            start_time = time.time()
            preds = model(batch_image_tensors.cuda())  # BDHW -> WBD
            result = codec.decode(preds.cpu().detach().numpy())
            time_consumed = time.time() - start_time

            print("max_width: {}, throughput: {} ms/img ".format(maxW,
                (time_consumed / args.batch_size) * 1000))
            print("predicted results: {}".format(result))


def preprocess_input(input_dir, height):
    img_list = []
    for img_name in os.listdir(input_dir):
        if img_name.endswith('png'):
            img_path = os.path.join(input_dir, img_name)
            src = cv2.imread(img_path)
            if len(src.shape) == 3:
                src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            ratio = float(src.shape[1]) / float(src.shape[0])
            th = height
            tw = int(th * ratio)
            rsz = cv2.resize(src, (tw, th),
                             fx=0, fy=0, interpolation=cv2.INTER_AREA)
            img_list.append(rsz)
    return img_list


def get_model_info(model_type):
    '''Get specific model information: model, characters'''
    model = None
    characters = ''
    chars_list_file = ''
    if model_type == 'hctr':
        model = hctr_model()
        chars_list_file = './data/handwritten_ctr_data/chars_list.txt'
    else:
        raise ValueError(
            'Model type: {} not supported'.format(model_type)
        )

    with open(chars_list_file, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            characters += line

    return model, model.img_height, characters


def benchmark(model, codec, args):
    AlignCollate_test = AlignCollate(imgH=model.img_height, PAD=model.PAD)
    test_dataset = ImageDataset(data_path=args.benchmark_mode,
                                img_shape=(1, model.img_height),
                                phase='test',
                                batch_size=args.batch_size)
    data_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers,
                                              collate_fn=AlignCollate_test,
                                              pin_memory=True)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    err_rate = AverageMeter()
    nchars = 0
    total = 0

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(data_loader):  # test/val_loader
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
                assert isinstance(pre, str), pre
                assert isinstance(tru, str), tru
                errs = editdistance.eval(pre, tru)
                total += errs
                nchars += len(tru)

            if nchars == 0:
                print('character length of labels is 0!')
                sys.exit()

            # compute error rate
            cur_err = total * 1.0 / nchars
            err_rate.update(cur_err, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('TEST: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Err {err_rate.val:.4f} ({err_rate.avg:.4f})\t'.format(
                          i, len(data_loader), batch_time=batch_time,
                          data_time=data_time, err_rate=err_rate))

    if nchars == 0:
        print('character length of labels is 0!')
        sys.exit()
    CER = total * 1.0 / nchars  # character error rate
    print('Total Test CER: {}'.format(CER))
    return 1.0 - CER


if __name__ == '__main__':
    test()
