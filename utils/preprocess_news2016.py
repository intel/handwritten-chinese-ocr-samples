"""
Apache v2 license
Copyright (C) <2018-2021> Intel Corporation
SPDX-License-Identifier: Apache-2.0
"""


import argparse
import itertools
import json
import multiprocessing
import os
import random
import re

from multiprocessing import Manager, Process
multiprocessing.set_start_method('spawn', True)


def build_argparser():
    parser = argparse.ArgumentParser(description='News2016 preprocessing script')
    args = parser.add_argument_group('Options')
    args.add_argument('-cf', '--corpus-file', type=str, metavar='PATH', required=True,
                      help='corpus json file downloaded from '
                           'https://github.com/brightmart/nlp_chinese_corpus')
    args.add_argument('-rf', '--result-file', type=str, metavar='PATH', required=True,
                      help='file name of preprocessed corpus')
    args.add_argument('-cd', '--chars-dict', type=str, metavar='PATH', required=True,
                      help='file name of valid character dictionary')
    args.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                      help='number of working processes')
    args.add_argument('-rm', '--remove-middle-files', action='store_true',
                      help='remove middle files during the preprocessing')
    return parser


class LineProcessor(object):
    def __init__(self, chars_dict):
        # Chinese characters
        self.re_zhcn = re.compile('[\u4e00-\u9fa5]+')
        # these symbols are kept
        self.re_kept = re.compile('[\u4e00-\u9fa5|?|。|,|;|!|、|:|\'|“|”]+')
        # split a line with these symbols
        self.re_sep = re.compile('[?|。|;|!|:|~]+')
        self.common_char = set()
        with open(chars_dict, mode='r') as f:
            for line in f.readlines():
                self.common_char.add(line.strip())

    def process(self, line):
        '''
        Step 1: full-width to half-width
        Step 2: filter unknown chars
        Step 3: insert blank between every two chars
        '''
        line = ''.join([Q2B(uchar) for uchar in line])
        line = ''.join(c for c in line if c in self.common_char)
        line = ' '.join(c for c in line)
        return line


def Q2B(uchar):
    ''' full-width to half-width '''
    inside_code = ord(uchar)
    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xfee0
    if (inside_code < 0x0020) or (inside_code > 0x7e):
        return uchar
    return chr(inside_code)


def extract(json_file, num_of_workers=8):
    '''
    json format example:
    {
     "news_id": "610130831", 
     "keywords": "导游，门票",
     "title": "故宫淡季门票40元 “黑导游”卖外地客140元",
     "desc": "近日有网友微....导游”确实存在。窗口出售",
     "source": "新华网",
     "time": "03-22 12:00", 
     "content": "近日有网友微.....渠道购买门票。"
    }
    '''
    # Only extract 'content' and save them to several files for multiprocessing
    file_paths = []
    file_handles = []
    for i in range(num_of_workers):
        file_paths.append(json_file + '.textline' + '.p' + str(i) + '.txt')
        file_handles.append(open(file_paths[-1], mode='w', encoding='utf8'))
    loop_file_handles = itertools.cycle(file_handles)

    with open(json_file, 'r', encoding='utf8') as jf:
        for fh in loop_file_handles:
            jline = jf.readline()
            if jline == '':
                break
            jdict = json.loads(jline)
            content = jdict['content'].strip()
            if content != '':
                fh.write(content + '\n')
    for fh in file_handles:
        fh.close()
    return file_paths


def preprocess_file(partial_corpus_file, chars_dict, filtered_file_paths):
    '''
    Preprocess part of corpus and filter unknown characters
    which are not in the chars_dict.txt file.
    '''
    processor = LineProcessor(chars_dict=chars_dict)
    file_path = partial_corpus_file + '.unk_chars_filtered.txt'
    file_handle = open(file_path, encoding='utf8', mode='w')

    with open(partial_corpus_file, 'r', encoding='utf8') as f:
        for line in f.readlines():
            res = processor.process(line)
            file_handle.write(res + '\n')
    file_handle.close()
    filtered_file_paths.append(file_path)


def main():
    args = build_argparser().parse_args()

    # check input files existance
    if not ((os.path.isfile(args.corpus_file)) and (os.path.isfile(args.chars_dict))):
        raise FileNotFoundError(
            "Files not found, expected corpus and dict file."
        )

    print('extracting...')
    extracted_file_paths = extract(args.corpus_file, num_of_workers=args.workers)

    process_list = []
    manager = Manager()
    filtered_file_paths = manager.list()

    print('processing files with multiprocessing...')
    for f in extracted_file_paths:
        p = Process(target=preprocess_file,
                    args=(f, args.chars_dict, filtered_file_paths))
        process_list.append(p)
        p.start()
    for i in range(len(extracted_file_paths)):
        process_list[i].join()

    print('merging processed files...')
    with open(args.result_file, mode='w') as rf:
        for file_path in filtered_file_paths:
            with open(file_path, mode='r') as f:
                for line in f.readlines():
                    rf.write(line)

    if args.remove_middle_files:
        for file_path in extracted_file_paths:
            os.remove(file_path)
        for file_path in filtered_file_paths:
            os.remove(file_path)
        print('middle files are removed')
    print('done!')


if __name__ == "__main__":
    main()
