"""
Apache v2 license
Copyright (C) <2018-2021> Intel Corporation
SPDX-License-Identifier: Apache-2.0
"""


import torch.nn as nn


class SELayer(nn.Module):
    """
    SENet Paper: https://arxiv.org/pdf/1709.01507.pdf
    Code: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
    """
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.dropout(out)
        return out


class ResNet(nn.Module):
    def __init__(self, in_channel, out_channel, block, num_blocks):
        super(ResNet, self).__init__()
        inout_channels = [int(out_channel / 8), # 64
                          int(out_channel / 4), #128
                          int(out_channel / 2), #256
                          int(out_channel / 1), #512
                          out_channel]          #512
        self.inplanes = int(out_channel / 8) #64

        self.conv0_1 = nn.Conv2d(in_channel, inout_channels[0], 3, 1, 1)
        self.bn0_1 = nn.BatchNorm2d(inout_channels[0])
        self.conv0_2 = nn.Conv2d(inout_channels[0], self.inplanes, 3, 1, 1)
        self.bn0_2 = nn.BatchNorm2d(self.inplanes)

        self.block1 = self._make_block(block, inout_channels[1], num_blocks[0])
        self.conv1 = nn.Conv2d(inout_channels[1], inout_channels[1], 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(inout_channels[1])

        self.block2 = self._make_block(block, inout_channels[2], num_blocks[1])
        self.conv2 = nn.Conv2d(inout_channels[2], inout_channels[2], 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(inout_channels[2])

        self.block3 = self._make_block(block, inout_channels[3], num_blocks[2])
        self.conv3 = nn.Conv2d(inout_channels[3], inout_channels[3], 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(inout_channels[3])

        self.block4 = self._make_block(block, inout_channels[4], num_blocks[3])
        self.conv4 = nn.Conv2d(inout_channels[4], inout_channels[4], 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(inout_channels[4])

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.9)

    def _make_block(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        blocks = []
        blocks.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            blocks.append(block(self.inplanes, planes))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv0_1(x)
        x = self.bn0_1(x)
        x = self.relu(x)
        x = self.conv0_2(x)
        x = self.bn0_2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.block1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout1(x)

        x = self.block2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout2(x)

        x = self.block3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout3(x)

        x = self.block4(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout4(x)

        return x


class hctr_model(nn.Module):
    def __init__(self):
        super(hctr_model, self).__init__()
        self.img_height = 128
        self.PAD = 'NormalizePAD'
        self.optimizer = 'SGD'
        self.pred = 'CTC'
        self.noutput = 1 + 7373 + 1 # 'blank' of CTC + num of characters + 'unknown'

        self.cnn = ResNet(1, 512, BasicBlock, [2, 4, 5, 1])
        self.linear = nn.Linear(2048, self.noutput)

    def forward(self, input):
        x = self.cnn(input)
        x = x.flatten(1, 2) # B[CH]W-> BDL
        x = x.permute(0, 2, 1) # BDL -> BLD
        x = self.linear(x)
        x = x.permute(1, 0, 2) # BLD -> LBD

        return x
