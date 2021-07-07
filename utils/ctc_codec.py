"""
Apache v2 license
Copyright (C) <2018-2021> Intel Corporation
SPDX-License-Identifier: Apache-2.0
"""


import itertools
import numpy as np
from scipy.special import log_softmax

NEG_INF = float('-inf')

class ctc_codec(object):
    """ Convert between text-label and text-index """

    def __init__(self, characters_str):
        # characters_str: set of the possible characters defined by users.
        self.chars_list = list(characters_str)

        self.dict = {}
        for i, char in enumerate(self.chars_list):
            # NOTE: 0 is reserved for 'blank' token required by CTCLoss
            self.dict[char] = i + 1

        # Insert '<blank>' token for CTCLoss (index 0)
        # Insert '<unknown>' token for supporting characters not in predefined chars_list
        self.characters = ['<blank>'] + self.chars_list + ['<unknown>']
        self.dict['<blank>'] = 0
        self.dict['<unknown>'] = len(self.characters) - 1

        self.ngram = None
        self.transformer = None
        self.lm_panelty = 2 # 2: ngram 0.8: transformer
        self.len_bonus = 5.8 # 5.8: ngram 4.8: transformer
        self.search_depth = 10
        self.beam_size = 10
        self.use_tfm_score = False
        self.use_tfm_pred = True
        self.skip_search = False
        self.use_beam_search = False

    def encode(self, text):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
        output:
            text: concatenated text index for CTCLoss.
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]
        text_label = ''.join(text)
        text_index = []
        for char in text_label:
            if char in self.chars_list:
                text_index.append(self.dict[char])
            else:
                # index of unknown character
                text_index.append(len(self.characters) - 1)

        return (np.array(text_index, dtype=np.int32), np.array(length, dtype=np.int32))

    def decode(self, preds):
        if self.use_beam_search:
            preds = log_softmax(preds, axis=2)
            return self.__cbs_skip__(preds) if self.skip_search else \
                   self.__cbs_full__(preds)
        return self.__greedy_search__(preds)

    def __greedy_search__(self, preds):
        """ convert text-index into text-label. """
        texts = []
        index = 0
        # Select max probabilty (greedy decoding) then decode index to character
        preds_index = np.argmax(preds, 2) # WBD -> WB
        preds_index = preds_index.transpose(1, 0).reshape(-1) # WB -> BW -> B*W
        preds_sizes = np.array([preds.shape[0]] * preds.shape[1])
        text_index = preds_index
        length = preds_sizes

        for l in length:
            t = text_index[index:index + l]

            # NOTE: t might be zero size
            if t.shape[0] == 0:
                continue

            char_list = []
            for i in range(l):
                # Remove repeated characters and blanks.
                if (t[i] != 0) and (t[i] != (len(self.characters) - 1)) and \
                   (not (i > 0 and t[i - 1] == t[i])):
                    char_list.append(self.characters[t[i]])
            text = ''.join(char_list)

            texts.append(text)
            index += l

        return texts

    def set_beam_search(self, skip_search=False, ngram_path='', tfm_path='',
                        lm_panelty=2, len_bonus=5.8, beam_size=10, search_depth=10,
                        use_tfm_score=False, use_tfm_pred=True,
                        use_openvino=False):
        self.use_beam_search = True
        self.lm_panelty = lm_panelty
        self.len_bonus = len_bonus
        self.beam_size = beam_size
        self.search_depth = search_depth
        self.use_tfm_pred = use_tfm_pred
        self.use_tfm_score = use_tfm_score
        self.skip_search = skip_search
        if use_tfm_pred or use_tfm_score:
            if not use_openvino:
                from .transformer_infer import TransformerWrapped
                self.transformer = TransformerWrapped(tfm_path)
            else:
                from .transformer_infer import TransformerOVIE
                self.transformer = TransformerOVIE(tfm_path)
        if not use_tfm_score:
            import kenlm
            self.ngram = kenlm.Model(ngram_path)

    def __cbs_skip__(self, preds):
        texts = []
        preds_size, batch_size, _ = preds.shape # WBD
        topk_idx = np.flip(np.argsort(preds, axis=2), axis=2)[:,:,:self.search_depth]
        prune_thresh = np.log(0.001) # TODO: parameter !
        for b in range(batch_size):
            top_line = [] # (greedy-decoded-character, time-step)
            top1_idx = topk_idx[:, b, 0]
            for t in range(preds_size):
                if (top1_idx[t] != 0) and \
                   (top1_idx[t] != len(self.characters) - 1) and \
                   (not (t > 0 and top1_idx[t-1] == top1_idx[t])):
                    top_line.append((self.characters[top1_idx[t]], t))
            # if top_line is none ?
            # TODO: suffix length: 4 ?
            end_step = (top_line[-1][1] + 4) if (top_line[-1][1] + 4) < preds_size else \
                preds_size
            kept_beams = [Beam()]
            for t in range(end_step):
                preds_at_t = preds[t, b, :]
                pruned = np.where(preds_at_t > prune_thresh)[0]
                # NOTE: Skip searching when only one prediction exceed prune_thresh
                # There might be better strategy here to balance accuracy and perf.
                if (pruned.shape[0] == 1):
                    pidx = pruned[0]
                    # Ignore the "unknown" character.
                    if pidx >= (len(self.characters) - 1):
                        continue
                    for kept_beam in kept_beams:
                        prefix = kept_beam.prefix
                        tail_idx = None if (prefix == '') else self.dict[prefix[-1]]
                        if pidx == 0:
                            # Prefix doesn't change, and only updates pb.
                            kept_beam.pb = kept_beam.prob() + preds_at_t[0]
                        elif pidx != tail_idx:
                            kept_beam.prefix += self.characters[pidx]
                            kept_beam.pnb = kept_beam.prob() + preds_at_t[pidx]
                            kept_beam.pb = NEG_INF # reset as it is an new beam
                        else: # pidx == tail_idx
                            if kept_beam.pb != NEG_INF:
                                # Not merge and include previous pb.
                                kept_beam.prefix += self.characters[pidx]
                                kept_beam.pnb = kept_beam.pb + preds_at_t[pidx]
                                kept_beam.pb = NEG_INF # reset as it is an new beam
                            else:
                                # Merge and include previous pnb.
                                kept_beam.pb = kept_beam.prob() + preds_at_t[0]
                                kept_beam.pnb = kept_beam.pnb + preds_at_t[pidx]
                else: # Do normal context beam search.
                    suffix = ''.join(c[0] for c in \
                        itertools.dropwhile(lambda tpl: tpl[1] <= t, top_line))[:4]
                    kept_beams = self.__context_beam_search__(kept_beams,
                        visual_candidates=pruned,
                        preds_at_t=preds_at_t,
                        suffix=suffix)
            texts.append(kept_beams[0].prefix)
        
        return texts

    def __cbs_full__(self, preds):
        texts = []
        preds_size, batch_size, _ = preds.shape # WBD
        topk_idx = np.flip(np.argsort(preds, axis=2), axis=2)[:,:,:self.search_depth]
        for b in range(batch_size):
            top_line = [] # (greedy-decoded-character, time-step)
            top1_idx = topk_idx[:, b, 0]
            for t in range(preds_size):
                # removing repeated characters and blanks.
                if (top1_idx[t] != 0) and \
                   (top1_idx[t] != (len(self.characters) - 1)) and \
                   (not (t > 0 and top1_idx[t-1] == top1_idx[t])):
                    top_line.append((self.characters[top1_idx[t]], t))
            # if top_line is none ?
            # TODO: suffix length: 4?
            end_step = (top_line[-1][1] + 4) if (top_line[-1][1] + 4) < preds_size else \
                       preds_size
            kept_beams = [Beam()]
            for t in range(end_step):
                suffix = ''.join(c[0] for c in \
                    itertools.dropwhile(lambda tpl: tpl[1] <= t, top_line))[:4]
                kept_beams = self.__context_beam_search__(kept_beams,
                    visual_candidates=topk_idx[t, b, :],
                    preds_at_t=preds[t, b, :],
                    suffix=suffix)
            texts.append(kept_beams[0].prefix)

        return texts

    def __context_beam_search__(self, input_beams, visual_candidates, preds_at_t, suffix):
        # Step 1: Combine visual and linguistic candidates at current time step.
        combined_condidates = []
        if self.use_tfm_pred:
            linguistic_candidates = self.transformer.next_k_words(
                [beam.prefix for beam in input_beams],
                k=self.search_depth,
                char_based=True)
            # input_beam.prefix == '', [[visual_candidates], [visual_candidates]]
            # input_beam.prefix != '', [[visual_candidates], [linguistic_candidates]]
            for i, beam in enumerate(input_beams):
                if beam.prefix != '':
                    combined_condidates.append(itertools.chain(visual_candidates,
                        [self.dict[x] for x in linguistic_candidates[i]]))
                else:
                    combined_condidates.append(visual_candidates)
        else:
            # without transformer prediction
            # [[visual_candidates], ..., [visual_candidates]]
            combined_condidates = itertools.repeat(visual_candidates, len(input_beams))

        # Step 2: Generate intermediate beams by extending input beams with candidates.
        gen_beams = {}
        for input_beam, candidates in zip(input_beams, combined_condidates):
            for idx in candidates:
                # Ignore the "unknown" character.
                if idx >= (len(self.characters) - 1):
                    continue

                prefix = input_beam.prefix
                p = preds_at_t[idx]
                if prefix not in gen_beams:
                    gen_beams[prefix] = Beam(prefix=prefix, pb=NEG_INF, pnb=NEG_INF)
                # Prefix doesn't change, and only updates pb.
                if idx == 0:
                    gen_beams[prefix].pb = np.logaddexp(gen_beams[prefix].pb,
                                                        input_beam.prob() + p)
                    continue
                
                # Extend the prefix with current candidate, and only updates pnb.
                tail_idx = None if (prefix == '') else self.dict[prefix[-1]]
                n_prefix = prefix + self.characters[idx]
                if n_prefix not in gen_beams:
                    gen_beams[n_prefix] = Beam(prefix=n_prefix, pb=NEG_INF, pnb=NEG_INF)
                if idx != tail_idx:
                    gen_beams[n_prefix].pnb = np.logaddexp(gen_beams[n_prefix].pnb,
                                                           input_beam.prob() + p)
                else: # idx == tail_idx
                    # Not merge and include previous pb.
                    gen_beams[n_prefix].pnb = np.logaddexp(gen_beams[n_prefix].pnb,
                                                           input_beam.pb + p)
                    # Merge and include previous pnb.
                    gen_beams[prefix].pnb = np.logaddexp(gen_beams[prefix].pnb,
                                                         input_beam.pnb + p)

        # Step 3: Socre all beams with language model and sort but keep top ones.
        output_beams = gen_beams.values()
        if self.use_tfm_score:
            tfm_batch = [(beam.prefix + suffix) for beam in output_beams]
            tfm_scores = self.transformer.score(tfm_batch, char_based=True)
            for i, beam in enumerate(output_beams):
                beam.pt = \
                    tfm_scores[i] * self.lm_panelty + len(beam.prefix) * self.len_bonus
        else:
            # Score with ngram model trained by kenlm.
            for beam in output_beams:
                sentence = ' '.join(char for char in (beam.prefix + suffix))
                ngram_score = self.ngram.score(sentence, eos=False)
                beam.pt = \
                    ngram_score * self.lm_panelty + len(beam.prefix) * self.len_bonus

        return sorted(output_beams,
                      key=lambda v: v.total(),
                      reverse=True)[:self.beam_size]


class Beam(object):
    def __init__(self, prefix='', pb=float(0), pnb=NEG_INF):
        # decoded prefix
        self.prefix = prefix
        # probability of ending with blank
        self.pb = pb
        # probability of ending with non-blank
        self.pnb = pnb
        # probability of language model
        self.pt = 0

    def prob(self):
        return np.logaddexp(self.pb, self.pnb)

    def total(self):
        return np.logaddexp(self.pb, self.pnb) + self.pt

    def __str__(self):
        return '[{}], {:.2f}, {:.2f}, {:.2f}'.format(self.prefix,
            self.pb, self.pnb, self.prob())
