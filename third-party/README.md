# Languange model training

## Corpus Preprocessing

Download the corpus "news2016zh" from [here](https://github.com/brightmart/nlp_chinese_corpus) and run the preprocess_news2016zh.py which includes:
1. Extract ```content``` only
2. Convert fullwidth characters into halfwidth, like ```，```to ```,```
3. Filter all the unknown characters out of chars_list.txt
4. Insert spaces between every two characters

## Train N-gram language model with [KenLM](https://github.com/kpu/kenlm)

### Installation

1. Clone and build, install other dependency if needed
   ```
   git clone https://github.com/kpu/kenlm
   cd kenlm
   mkdir build && cd build
   cmake ..
   make
   ```
2. Install Python module
   ```
   pip install https://github.com/kpu/kenlm/archive/master.zip
   ```

### Training

Detailed usage about kenlm can be found [here](https://kheafield.com/code/kenlm/).
1. Train a 5-gram model ```model.arpa``` with preprocessed corpus file ```corpus.txt```
   ```
   ./kenlm/build/bin/lmplz -o 5 -S 80% <corpus.txt >model.arpa
   ```
   Explanation of arguments:  
   ```-o``` is required. Denotes the order of the language model to estimate.  
   ```-S``` is recommended. Denotes memory to use. % for percentage of physical memory.

2. Convert the ARPA to binary file for fast loading
   ```
   ./kenlm/build/bin/build_binary model.arpa model.bin
   ```


## Train Transformer-based language model with [Fairseq](https://github.com/pytorch/fairseq/).

### Installation  
Please refer to more details from [official instruction of fairseq installation.](https://github.com/pytorch/fairseq/)
   ```
   git clone https://github.com/pytorch/fairseq
   cd fairseq
   git checkout release/v0.10.2
   pip install --editable ./
   ```

### Preprocessing
   ```
   TEXT=<directory-of-the-preprocessed-corpus>
   fairseq-preprocess \
      --only-source \
      --trainpref $TEXT/train.txt \
      --validpref $TEXT/valid.txt \
      --testpref $TEXT/hwdb2x_test_page_gt_space.txt \
      --destdir data-bin/news2016zh-hwdb \
      --workers 20
   ```

### Training
Two Titan V GPUs are used for training this language model. See [official nerual language model example](https://github.com/pytorch/fairseq/tree/master/examples/language_model) for more details.
   ```
   fairseq-train \
      --task language_modeling \
      data-bin/news2016zh-hwdb \
      --save-dir checkpoints/news2016zh-hwdb \
      --arch transformer_lm \
      --share-decoder-input-output-embed \
      --dropout 0.1 \
      --optimizer adam \
      --adam-betas '(0.9, 0.98)' \
      --weight-decay 0.01 \
      --clip-norm 0.0 \
      --lr 0.0005 \
      --lr-scheduler inverse_sqrt \
      --warmup-updates 400 \
      --warmup-init-lr 1e-07 \
      --tokens-per-sample 512 \
      --sample-break-mode none \
      --max-tokens 32768 \
      --update-freq 16 \
      --fp16 \
      --max-update 100000 \
   ```

### Testing
For this Handwritten Chinese OCR task, selected news2016zh corpus, and the default transformer_lm arch, PPL=29.xx on hwdb2x_test_page_gt_space.txt is good enough.
   ```
   fairseq-eval-lm \
      data-bin/data-bin/news2016zh-hwdb \
      --path checkpoints/news2016zh-hwdb/checkpoint_last.pt \
      --max-sentences 2 \
      --tokens-per-sample 512 \
      --context-window 400
   ```
