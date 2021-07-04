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
import numpy as np
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
                      type=str, required=True, choices=['hctr'],
                      help='target model for different languages and scenarios')
    args.add_argument('-f', '--model-file', dest='model_file',
                      type=str, metavar='PATH', required=True,
                      help='path to best model file')
    args.add_argument('-i', '--input', dest='input',
                      type=str, metavar='PATH', required=True,
                      help='path to input image or testset')
    args.add_argument('-b', '--batch-size', dest='batch_size',
                      type=int, metavar='N', default=1,
                      help='mini-batch size')
    args.add_argument('--gpu', type=int, default=None,
                      help='GPU id to use.')
    args.add_argument('-bm', '--benchmark-mode', dest='benchmark_mode',
                      action='store_true',
                      help='enable to benchmark on input testset.')
    args.add_argument('-dm', '--decode-method', dest='decode_method',
                      type=str, default='beam-search',
                      choices=['greedy-search', 'beam-search'],
                      help='method to decode the CTC output.')
    args.add_argument('-ss', '--skip-search', dest='skip_search',
                      action='store_true',
                      help='whether skip high confidence characters ' +
                           'when using beam search.')
    args.add_argument('-kp', '--kenlm-path', dest='kenlm_path',
                      type=str, metavar='PATH',
                      help='ngram model for scoring in beam search.')
    args.add_argument('-utp', '--use-tfm-pred', dest='use_tfm_pred',
                      action='store_true',
                      help='use transformer for candidates prediction.')
    args.add_argument('-tp', '--transformer-path', dest='tfm_path',
                      type=str, metavar='DIR',
                      help='path to transformer language model.')
    args.add_argument('-uts', '--use-tfm-score', dest='use_tfm_score',
                      action='store_true',
                      help='use transformer for scoring in beam search.')
    args.add_argument('-uov', '--use-openvino', dest='use_openvino',
                      action='store_true',
                      help='use openvino to do transformer ' +
                           'model inference during beam search.')
    args.add_argument('-bs', '--beam-size', dest='beam_size',
                      type=int, default=10,
                      help='beam size for beam search.')
    args.add_argument('-sd', '--search-depth', dest='search_depth',
                      type=int, default=10,
                      help='search depth (top-k) for beam search.')
    args.add_argument('-lp', '--lm-panelty', dest='lm_panelty',
                      type=float, default=0.8,
                      help='panelty of language model for sentences scoring.')
    args.add_argument('-lb', '--len-bonus', dest='len_bonus',
                      type=float, default=4.8,
                      help='length bonus for sentences scoring.')
    args.add_argument('-jw', '--workers',
                      type=int, metavar='N', default=4,
                      help='number of data loading workers in benchmark mode.')
    args.add_argument('-tv', '--test-verbose', dest='test_verbose',
                      action='store_true',
                      help='print result during model testing.')
    args.add_argument('-pf', '--print-freq', dest='print_freq',
                      type=int, metavar='N', default=100,
                      help='log print frequency during model testing.')
    ###########################################################################
    # subgroup of parameters for hyper-param tunning only.
    args.add_argument('-gs', '--grid-search', action='store_true',
                      help='use grid search for lm_panelty and len_bonus.')
    args.add_argument('-al', '--alpha-lower', type=float, default=0.7,
                      help='alpha(lm_panelty) lower bound')
    args.add_argument('-au', '--alpha-upper', type=float, default=1.1,
                      help='alpha(lm_panelty) upper bound')
    args.add_argument('-ac', '--alpha-count', type=int, default=10,
                      help='alpha(lm_panelty) count')
    args.add_argument('-bl', '--beta-lower', type=float, default=4.2,
                      help='beta(len_bonus) lower bound')
    args.add_argument('-bu', '--beta-upper', type=float, default=6.6,
                      help='beta(len_bonus) upper bound')
    args.add_argument('-bc', '--beta-count', type=int, default=25,
                      help='beta(len_bonus) count')

    return parser


def test(args):
    if os.path.isfile(args.model_file) is False:
        raise FileNotFoundError(
            'No model file found at: {}'.format(args.model_file)
        )

    if (os.path.isdir(args.input) or os.path.isfile(args.input)) is False:
        raise FileNotFoundError(
            'Input is not found, expected file or folder.'
        )

    if args.gpu is not None:
        print('Use GPU: {} for testing'.format(args.gpu))

    # create model specific info
    model, img_height, characters = get_model_info(args.model_type)
    # depends on using ctc or attension
    codec = ctc_codec(characters)
    if args.decode_method == 'beam-search':
        codec.set_beam_search(args.skip_search,
                              ngram_path=args.kenlm_path,
                              tfm_path=args.tfm_path,
                              lm_panelty=args.lm_panelty,
                              len_bonus=args.len_bonus,
                              beam_size=args.beam_size,
                              search_depth=args.search_depth,
                              use_tfm_score=args.use_tfm_score,
                              use_tfm_pred=args.use_tfm_pred,
                              use_openvino=args.use_openvino
        )

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # Use CPU for testing
        model = model.cpu()

    # Load pre-trained model parameters
    print('=> loading model file: {}'.format(args.model_file))
    checkpoint = torch.load(args.model_file, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    # Switch to evaluate mode
    model.eval()

    if args.benchmark_mode == True:
        return benchmark(model, codec, args)

    # Preprocess the input images if needed
    input_list = preprocess_input(args.input, img_height)

    batch_num = len(input_list) // args.batch_size
    with torch.no_grad():
        for i in range(batch_num):
            if i % 1 == 0:
                print('batch {} is being processed...'.format(i))

            maxW = 0
            batch_images = input_list[
                i * args.batch_size: (i+1) * args.batch_size
            ]
            padded_images = []
            for image in batch_images:
                h, w = image.shape
                if w > maxW:
                    maxW = w
            for image in batch_images:
                image = image[:, :, None]
                transform = NormalizePAD((1, img_height, maxW))
                padded_images.append(transform(image))

            batch_image_tensors = torch.cat(
                [t.unsqueeze(0) for t in padded_images], 0
            )

            # Compute and decode output
            start_time = time.time()
            if args.gpu is not None:
                preds = model(batch_image_tensors.cuda()) # BDHW -> WBD
            else:
                preds = model(batch_image_tensors)
            result = codec.decode(preds.cpu().detach().numpy())
            time_consumed = time.time() - start_time

            print('max_width: {}, throughput: {} ms/img'.format(maxW,
                (time_consumed / args.batch_size) * 1000))
            print('predicted results: {}'.format(result))

    return None


def preprocess_input(input, height):
    img_list = []

    def read_resize_image(img_path, height):
        src = cv2.imread(img_path)
        if len(src.shape) == 3:
            src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        ratio = float(src.shape[1]) / float(src.shape[0])
        th = height
        tw = int(th * ratio)
        rsz = cv2.resize(src, (tw, th),
            fx=0, fy=0, interpolation=cv2.INTER_AREA)
        return rsz

    if os.path.isfile(input):
        img_data = read_resize_image(input, height)
        img_list.append(img_data)
    else: # folder
        for img_name in os.listdir(input):
            img_path = os.path.join(input, img_name)
            img_data = read_resize_image(img_path, height)
            img_list.append(img_data)

    return img_list


def benchmark(model, codec, args):
    if not os.path.isdir(args.input):
        raise AssertionError(
            'Input should be a folder under benchmark mode.'
        )
    AlignCollate_test = AlignCollate(imgH=model.img_height, PAD=model.PAD)
    test_dataset = ImageDataset(data_path=args.input,
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
        for i, (input, target) in enumerate(data_loader): # test/val_loader
            # measure data loading time
            data_time.update(time.time() - end)

            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            preds = model(input)
            result = codec.decode(preds.cpu().detach().numpy())

            for j, (pre, tru) in enumerate(zip(result, target)):
                if args.test_verbose:
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
    return CER


def get_model_info(model_type):
    '''Get specific model information: model, characters'''
    model = None
    characters = ''
    chars_list_file = ''
    if model_type == 'hctr':
        model = hctr_model()
        chars_list_file = \
            './data/handwritten_ctr_data/chars_list.txt'
    else:
        raise ValueError(
            'Model type: {} not supported'.format(model_type)
        )

    with open(chars_list_file, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            characters += line

    return model, model.img_height, characters


if __name__ == '__main__':
    args = build_argparser().parse_args()
    if args.grid_search == False:
        test(args)
    else:
        # use grid search to find the best lm_panelty and len_bonus
        # recommendation for short execution time of each iteration
        # small set of validation data (e.g. 10%)
        # small beam size and search depth (e.g. 5)
        # disable use_tfm_pred
        if not (args.benchmark_mode == True):
            raise AssertionError(args.benchmark_mode)
        alpha = np.linspace(args.alpha_lower,
            args.alpha_upper, args.alpha_count)
        beta = np.linspace(args.beta_lower,
            args.beta_upper, args.beta_count)

        min_cer = 1.0
        min_params = (0, 0)
        for a in alpha:
            for b in beta:
                print('searching with a:{}, b:{}, '
                      'min params:{}, min cer:{}'
                      .format(a, b, min_params, min_cer)
                )
                args.lm_panelty = a
                args.len_bonus = b
                cer = test(args)

                if cer < min_cer:
                    min_cer = cer
                    min_params = (a, b)

        print('min params:{}, min cer: {}'
            .format(min_params, min_cer)
        )
