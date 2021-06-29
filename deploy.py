"""
Apache v2 license
Copyright (C) <2018-2021> Intel Corporation
SPDX-License-Identifier: Apache-2.0
"""


from __future__ import print_function
import os
import sys
import time
import logging as log
from argparse import ArgumentParser, SUPPRESS

import cv2
import numpy as np
from openvino.inference_engine import IECore

from utils.ctc_codec import ctc_codec

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', "--help", action='help', default=SUPPRESS,
                      help="Show this help message and exit")
    args.add_argument('-lang', '--language', type=str, required=True, choices=['hctr'],
                      help="Required. Target model for different languages")
    args.add_argument("-m", "--model", type=str, required=True,
                      help="Required. Path to an .xml file with a trained model")
    args.add_argument("-i", "--input", type=str, required=True,
                      help="Required. Path to an image file")
    args.add_argument("-d", "--device", type=str, default="CPU",
                      help="Optional. Specify the target device to infer on; "
                           "CPU, GPU, FPGA, HDDL, MYRIAD or HETERO: is acceptable. "
                           "The sample will look for a suitable plugin for device specified. "
                           "Default value is CPU")
    args.add_argument("-ni", "--number_iter", type=int, default=20,
                      help="Optional. Number of inference iterations")
    args.add_argument('-dm', '--decode-method', type=str, dest='method',
                      choices=['greedy-search', 'beam-search'],
                      default='greedy-search')
    args.add_argument('-ss', '--skip-search', action='store_true', dest='skip_search',
                      help='whether skip high confidence characters when using beam search')
    args.add_argument('-kp', '--kenlm-path', metavar='PATH', type=str, dest='kenlm_path',
                      help='ngram model for scoring in beam search')
    args.add_argument('-utp', '--use-tfm-pred', action='store_true', dest='use_tfm_pred',
                      help='use transformer for candidates prediction')
    args.add_argument('-uts', '--use-tfm-score', action='store_true', dest='use_tfm_score',
                      help='use transformer for scoring in beam search')
    args.add_argument('-uov', '--use-openvino', action='store_true', dest='use_openvino',
                      help='use openvino for transformer inference in beam search')
    args.add_argument('-tfm', '--transformer-model', metavar='DIR', type=str, dest='tfm_path',
                      help='path to transformer model')
    args.add_argument('-bs', '--beam-size', type=int, default=10, dest='beam_size',
                      help='beam size in beam search')
    args.add_argument('-sd', '--search-depth', type=int, default=10, dest='search_depth',
                      help='search depth (top-k) in beam search')
    args.add_argument('-lp', '--lm-panelty', default=1.9, type=float, dest='lm_panelty',
                      help='panelty of language model scoring in beam search')
    args.add_argument('-lb', '--len-bonus', default=5.7, type=float, dest='len_bonus',
                      help='length bonus of scoring in beam search')

    return parser


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    # Prepare the language specific information, characters list and codec method
    model_characters = get_model_characters(args.language)
    codec = ctc_codec(model_characters)

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
                              use_openvino=args.use_openvino
        )

    # Plugin initialization for specified device and load extensions library if specified
    ie = IECore()
    # Read IR
    log.info("Loading network files...\n\t{}\n\t{}".format(model_xml, model_bin))
    net = ie.read_network(model=model_xml, weights=model_bin)

    if not (len(net.input_info.keys()) == 1):
        raise AssertionError("Sample supports only single input topologies")
    if not (len(net.outputs) == 1):
        raise AssertionError("Sample supports only single output topologies")

    log.info("Preparing input/output blobs...")
    input_blob = next(iter(net.input_info))
    output_blob = next(iter(net.outputs))

    n, c, h, w = net.input_info[input_blob].input_data.shape
    log.info("Loading model to the plugin...")
    exec_net = ie.load_network(network=net, device_name=args.device)

    # Read and pre-process input images (NOTE: image by image ONLY)
    if os.path.isfile(args.input):
        input_image = preprocess_input(args.input, height=h, width=w)
        # Start sync inference
        log.info("Starting inference ({} iterations)...".format(args.number_iter))
        infer_time = []
        for i in range(args.number_iter):
            t0 = time.time()
            res = exec_net.infer(inputs={input_blob: input_image})
            res = res[output_blob]
            res = codec.decode(res)
            infer_time.append((time.time() - t0) * 1000)
            log.info("Showing the prediction...\nfile:\t{}\npred:\t{}"
                .format(args.input, res)
            )
        log.info("Average throughput: {} ms".format(
            np.average(np.asarray(infer_time)))
        )
    else:
        for img_file in os.listdir(args.input):
            input_image = preprocess_input(
                os.path.join(args.input, img_file),
                height=h,
                width=w
            )
            res = exec_net.infer(inputs={input_blob: input_image})
            res = res[output_blob]
            res = codec.decode(res)
            log.info("Showing the prediction...\nfile:\t{}\npred:\t{}"
                .format(img_file, res)
            )

    sys.exit()


def preprocess_input(image_file, height, width):
    '''Transform the input image to the format of model required: fix height and width'''
    src = cv2.imread(image_file)
    if len(src.shape) == 3:
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # Adjust height if needed
    h, w = src.shape
    if h != height:
        ratio = w / h
        th = height
        tw = int(height * ratio)
        src = cv2.resize(src, (tw, th), interpolation=cv2.INTER_AREA)
    # Apply padding if needed
    h, w = src.shape
    pad_img = np.ones((h, width), dtype=np.uint8) * 255
    if w >= width:
        # ignore the right part which exceeds the model required
        pad_img = src[:, :width]
    else:
        pad_img[:, :w] = src
        # pad with the right border
        pad_img[:, w:] = np.tile(src[:, [-1]], width - w)

    # Normalize
    norm_img = (pad_img - 127.5) / 127.5

    return norm_img[None, None, :, :]


def get_model_characters(language):
    '''Get specific model information: characters list'''
    characters = ''
    chars_list_file = ''
    if language == 'hctr':
        chars_list_file = './data/handwritten_ctr_data/chars_list.txt'
    else:
        raise ValueError(
            'Model type: {} not supported'.format(language)
        )

    with open(chars_list_file, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            characters += line

    return characters


if __name__ == '__main__':
    main()
