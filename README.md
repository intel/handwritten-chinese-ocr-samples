# OCRtextline
This project is aim to create a simple and unified text line recognition solution using CNN+CTC method for different handwriting recognition scenarios including: 
* HCTR (offline handwritten Chinese text recognition)
* HJTR (offline handwritten Japanese text recognition)
* HETR (offline handwritten Englist text recognition)

| Scenerios | Training data | Testing data | Character accuracy rate w/o LM | Character accuracy rate w/ LM |
| --- | --- | --- | --- | --- |
| OCRtextline-HCTR | CASIA-HWDB2.x/1.x | ICDAR2013 competition set | 93.68% (previous [91.58%](https://arxiv.org/abs/1812.09809)) | 97.51% (previous [96.83%](https://arxiv.org/abs/1812.09809)) |
| OCRtextline-HJTR | Kondate trainset  | Kondate testset           | 98.43% (previous [96.35%](https://ieeexplore.ieee.org/document/8563229/)) | - |
## Prerequisites
* [PyTorch](https://github.com/pytorch/pytorch) 1.7.1
* [Anaconda](https://www.anaconda.com/distribution/) with Python 3.7 or above
* NVIDIA CUDA 10.0 or above
* PyTorch Binding for [warp-ctc](http://github.com/SeanNaren/warp-ctc.git)

## Model Training
### Dataset and Structure
Restricted dataset for research only for different scenarios:
* [CASIA-HWDB](http://www.nlpr.ia.ac.cn/databases/handwriting/Home.html)
* [Kondate](http://web.tuat.ac.jp/~nakagawa/database/index.html)
* [IAM](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)

All above datasets follow an uniformed structure to represent the data for train/val/test. The layout of train/val/test_img_id_gt.txt are designed as one sample per line, which uses a comma to separate the image id and its label:
```
img_id_1,text_1
img_id_2,text_2
img_id_3,text_3
...
```
Character set of corresponding dataset is represented in a text file. The format of the chars_list.txt is also designed as one character per line:
```
character_1
character_2
character_3
...
```
### Training
Note that, before training with a target dataset, all the gray-scale images should be resized to fixed-height (e.g. 128) with fixed-ratio.
```
python main.py -m [hctr|hjtr] -d <dataset_directory> -b 8 -pf 100 -lr 1e-4 --gpu 0
```
### Customizing
1. Create new folder under data/ and characters list for new dataset following the above representation;
2. Reuse or redefine the model topology under models/ for new application;
3. Train and tune the model.

## Model Test
After the training, the model can then be tested with test set or single image for benchmarking. In order to get the output of CTC-based text recognition, there is one additional step called decoding which is critial to final accuracy.
```
python test.py -m [hctr|hjtr] -f <trained_model_path> \
               -i <input_image> (or -ts <test_set_directory>) \
               -dm [greedy-search|beam-search] \
               ...
```
### Decoding with greedy-search
This is a basic and default decoding method, and it only takes the maximum probalitiy at each output step.
### Decoding with beam-search (with language model)
To further improve the accuracy, a specific language model (n-gram or transformer-based) can be introduced to work with the beam-search decoding. There are two major strategies provided to configurate the beam-search.
* For the best accuracy, use: ```--decode-method beam-search --use-tfm-pred --transformer-path <path>```
* For the shortest latency, use: ```--decode-method beam-search --skip-search --kenlm-path <path>```

See the instructions under third-party/ folder to train a specific n-gram or transformer-based language model.

## Model Deployment
1. Convert the PyTorch model to onnx with export_onnx.py
2. Convert the onnx model to OpenVINO IR xml and bin (optional)
3. Inference with OpenVINO on target platform with predict_openvino.py

## Acknowledgments
* This project created the main.py from [PyTorch/examples/imagenet](https://github.com/pytorch/examples/tree/master/imagenet)
* This project integrated the pytorch binding [warp-ctc](https://github.com/SeanNaren/warp-ctc) shared by **Sean Naren**
