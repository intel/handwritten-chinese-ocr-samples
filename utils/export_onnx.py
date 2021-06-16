"""
Apache v2 license
Copyright (C) <2018-2021> Intel Corporation
SPDX-License-Identifier: Apache-2.0
"""


import os
import sys
import argparse
import numpy as np

import torch


parser = argparse.ArgumentParser(description='PyTorch to ONNX convertion')
parser.add_argument('-m', '--model-type', type=str, required=True,
                    choices=['hctr', 'tfm'],
                    help='target model for different languages and usages')
parser.add_argument('-f', '--model-file', type=str, required=True, metavar='PATH',
                    help='path to model file')
parser.add_argument('-w', '--input-width', default=2000, type=int, metavar='N',
                    help='input image width')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    help='batch size of input')

args = parser.parse_args()
if os.path.isfile(args.model_file) is False:
    print('No model file found at: {}'.format(args.model_file))
    sys.exit()

if args.model_type == 'hctr':
    from models.handwritten_ctr_model import hctr_model
    model = hctr_model()
    checkpoint = torch.load(args.model_file, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    print(model)

    # export model to onnx
    model_dummy_input = torch.randn(
        args.batch_size,
        1, # channel
        model.img_height,
        args.input_width
    ).cpu()

    torch.onnx.export(
        model,
        model_dummy_input,
        f'model_{args.model_type}_pt.onnx',
        verbose=True,
        input_names=['actual_input'],
        output_names=['output']
    )

elif args.model_type == 'tfm':
    from fairseq.models.transformer_lm import TransformerLanguageModel
    model_path = os.path.dirname(args.model_file)
    model_name = 'checkpoint_best.pt'
    transformer = TransformerLanguageModel.from_pretrained(
        model_path, model_name
    ).eval()
    decoder = transformer.models[0].decoder
    print(decoder)
    decoder.prepare_for_onnx_export_()

    batch_size = 1 # fix batch size for OpenVino inference
    token_len = 64 # fix input length too
    model_dummy_input = torch.zeros(
        size=(batch_size, token_len),
        dtype=torch.long
    )

    # RuntimeError 1:
    # Only tuples, lists and Variables supported as JIT inputs/outputs.
    # WORKAROUND: Remove the "extra" output of the forward func of
    # class TransformerDecoder in fairseq/models/transformer.py.
    # return x, extra -> return x

    # RuntimeError 2:
    # Exporting the operator triu to ONNX opset version 11 is not supported.
    # https://github.com/pytorch/pytorch/issues/32968
    # WORKAROUND begin
    def triu_onnx(x, diagonal=0):
        l = x.shape[0]
        arange = torch.arange(l, device=x.device)
        mask = arange.expand(l, l)
        arange = arange.unsqueeze(-1)
        if diagonal:
            arange = arange + diagonal
        mask = mask >= arange
        return x.masked_fill(mask == 0, 0)

    torch.triu = triu_onnx
    # WORKAROUND end

    torch.onnx.export(
        decoder,
        model_dummy_input,
        f'model_{args.model_type}_pt.onnx',
        verbose=True,
        input_names=['actual_tokens'],
        output_names=['output'],
        opset_version=11
    )

else:
    raise ValueError(
        'Model type: {} not supported'.format(args.model_type)
    )


# Next: convert to OpenVINO IR with model optimizer
# mo_onnx.py --input_model *.onnx --data_type [FP16/FP32]
