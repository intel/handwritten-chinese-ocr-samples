"""
Apache v2 license
Copyright (C) <2018-2021> Intel Corporation
SPDX-License-Identifier: Apache-2.0
"""


import os
import numpy as np
from typing import List
from scipy.special import log_softmax

import torch


# Depend on the library: fairseq from https://github.com/pytorch/fairseq
class TransformerWrapped(object):
    def __init__(self, model_path: str):
        from fairseq.models.transformer_lm import TransformerLanguageModel
        model_name = 'checkpoint_best.pt'
        model_file = os.path.join(model_path, model_name)
        dict_file  = os.path.join(model_path, 'dict.txt')
        if not (
            (os.path.isfile(model_file)) and \
            (os.path.isfile(dict_file))
        ):
            raise FileNotFoundError(
                "Files not found, expected model and dict file."
            )

        tfm_lm = TransformerLanguageModel.from_pretrained(
            model_path,
            model_name
        ).eval().cuda() # TODO: if no cuda ?

        self.decoder = tfm_lm.models[0].decoder
        # TODO: the reason why need to reimplement tokenizer
        self.tokenizer = Tokenizer(dict_file)
        self.log_softmax = torch.nn.LogSoftmax(dim=2)

    def score(self, sentences: List[str], char_based=False):
        # TODO: why there are sentences with length > 200 (beam size = depth size = 10) ?
        tokens = self.tokenizer.tokenize(sentences, char_based=char_based)
        tokens = torch.tensor(tokens, dtype=torch.long).cuda()
        preds = self.decoder.forward(tokens)[0] # [B][L][D]
        preds = self.log_softmax(preds)
        # tokens[:, 1:] is the target of tokens[:, :-1]
        # TODO: any other similiar gather func with numpy ?
        positional_scores = preds.gather(
            index=tokens[:, 1:].unsqueeze(-1),
            dim=2
        ).cpu()

        scores = np.zeros((len(sentences)))
        for i, sentence in enumerate(sentences):
            scores[i] = torch.sum(
                positional_scores[i, :len(sentence), :]
            )

        return scores

    def next_k_words(self, sentences: List[str], k: int, char_based=False):
        tokens = self.tokenizer.tokenize(sentences, char_based=char_based)
        tokens = torch.tensor(tokens, dtype=torch.long).cuda()
        preds = self.decoder.forward(tokens)[0].cpu() # [B][L][D]
        _, topk_idx = torch.topk(preds, dim=2, k=k)

        results = []
        for i, sentence in enumerate(sentences):
            results.append(
                self.tokenizer.decode(
                    list(topk_idx[i, len(sentence), :])
                )
            )

        return results


# Depend on the inference-engine of OpenVINO toolkit
# https://docs.openvinotoolkit.org/latest/index.html
class TransformerOVIE(object):
    def __init__(self, tfm_path: str):
        from openvino.inference_engine import IECore
        model_xml = os.path.join(tfm_path, 'model_tfm_pt.xml')
        model_bin = os.path.join(tfm_path, 'model_tfm_pt.bin')
        dict_txt  = os.path.join(tfm_path, 'dict.txt')
        if not (
            (os.path.isfile(model_xml)) and \
            (os.path.isfile(model_bin)) and \
            (os.path.isfile(dict_txt))
        ):
            raise FileNotFoundError(
                "Files not found, expected .xml, .bin and dict.txt"
            )

        ie = IECore()
        net = ie.read_network(model=model_xml, weights=model_bin)
        self.tfm = ie.load_network(network=net, device_name='CPU')

        self.input_blob = next(iter(self.tfm.input_info))
        self.output_blob = next(iter(self.tfm.outputs))
        self.input_shape = \
            self.tfm.input_info[self.input_blob].input_data.shape
        self.tokenizer = Tokenizer(dict_txt)

    def score(self, sentences: List[str], char_based=False):
        batch_size = self.input_shape[0]
        max_char_num = self.input_shape[1]
        scores = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:][:batch_size]
            while len(batch) < batch_size:
                batch.append('')

            input_data = self.tokenizer.tokenize(
                batch,
                char_based=char_based,
                fixed_len=max_char_num
            )

            pred = self.tfm.infer(
                inputs={self.input_blob: input_data}
            )[self.output_blob]

            probs = log_softmax(pred, axis=2)
            positional_scores = np.take_along_axis(
                probs[:, :-1],
                np.expand_dims(
                    input_data[:, 1:],
                    axis=-1
                ),
                axis=2
            )
            for j, b in enumerate(batch):
                scores.append(
                    np.sum(positional_scores[j, 0:len(b)])
                )

        return scores[:len(sentences)]

    def next_k_words(self, sentences: List[str], k: int, char_based=False):
        batch_size = self.input_shape[0]
        max_char_num = self.input_shape[1]
        chars = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:][:batch_size]
            while len(batch) < batch_size:
                batch.append('')

            input_data = self.tokenizer.tokenize(
                batch,
                char_based=char_based,
                fixed_len=max_char_num
            )

            pred = self.tfm.infer(
                inputs={self.input_blob: input_data}
            )[self.output_blob]

            for j, s in enumerate(batch):
                topk = np.argpartition(
                    -pred[j, len(s), :],
                    kth=k
                )
                chars.append(
                    self.tokenizer.decode(topk[:k])
                )

        return chars[:len(sentences)]


class Tokenizer(object):
    def __init__(self, dict_file):
        self.indices = {}
        self.symbols = ['<s>', '<pad>', '</s>', '<unk>']
        self.sos_index = 0
        self.eos_index = 2
        self.unk_index = 3

        with open(dict_file, mode='r') as f:
            lines = f.readlines()
            for line in lines:
                try:
                    word, field = line.rstrip().rsplit(" ", 1)
                    self.indices[word] = len(self.symbols)
                    self.symbols.append(word)
                except ValueError:
                    raise ValueError(
                        "Incorrect format, expected '<token> <cnt>'"
                    )

    def tokenize(self, sentences: List[str], char_based=False, fixed_len=-1):
        list_of_chars = [
            list(s) if char_based else s.split() for s in sentences
        ]
        max_len = fixed_len if fixed_len > 0 else \
            len(max(sentences, key=len)) + 1
        ids = np.full(
            (len(sentences), max_len),
            fill_value=self.eos_index
        )

        for i, chars in enumerate(list_of_chars):
            # NOTE: Adding sos_index at the beginning of sentences
            # and eos_index at the end of other shorter sentences
            # IS BETTER (experimentally)
            # than only adding eos_index at the end.
            ids[i, 0] = self.sos_index
            ids[i, 1:len(chars)+1] = list(
            #ids[i, :len(chars)] = list(
                map(lambda c: self.indices[c] if c in self.indices else
                    self.unk_index, chars
                )
            )

        return ids

    def decode(self, tokens: List[int]):
        return [
            self.symbols[x] for x in tokens if x > 3
        ]
