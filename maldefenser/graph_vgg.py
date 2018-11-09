#!/usr/bin/python3.7
import glog as log
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from typing import Dict, List
from torch.autograd import Variable
"""
VGG for the output of graph convolution layers,
most of the code is borrowed from
https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""

ModelUrls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
}

VggCfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vggg': [16, 'M', 32, 32, 'M', 64, 64, 'M'],
}


class GraphVgg(nn.Module):
    def __init__(self, convLayers, featDim, hidden=128, numClasses=9,
                 dropOutRate=0.4, initWeights=True):
        super(GraphVgg, self).__init__()
        self.convLayers = convLayers
        self.classifier = nn.Sequential(
            nn.Linear(featDim, hidden),
            nn.ReLU(True),
            nn.Dropout(p=dropOutRate),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Dropout(p=dropOutRate),
            nn.Linear(hidden, numClasses),
        )
        self.nllLoss = nn.NLLLoss()
        if initWeights:
            self._initialize_weights()

    def forward(self, x, y=None):
        x = self.convLayers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        predProb = F.softmax(x, dim=1)
        logits = F.log_softmax(x, dim=1)

        if y is not None:
            loss = self.nllLoss(logits, y)

            pred = logits.data.max(1)[1]
            correct = pred.eq(y.data.view_as(pred))
            accu = (correct.sum().item()) / float(correct.size(0))
            return loss, accu, pred
        else:
            log.debug('[MLPRegression] None label, return only predictions.')
            return predProb

    def print_result_dict(self):
        pass

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def makeVggLayers(cfg: List[str], inputDims: List[int], batchNorm=False):
    layers = []
    hout, wout, channels = inputDims
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=3, stride=2)]
            hout = (hout + 2 * 0 - (3 - 1) - 1) // 2 + 1
            wout = (wout + 2 * 0 - (3 - 1) - 1) // 2 + 1
        else:
            conv2d = nn.Conv2d(channels, v, kernel_size=3, padding=1)
            if batchNorm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            channels = v
            hout = (hout + 2 - (3 - 1) - 1) + 1
            wout = (wout + 2 - (3 - 1) - 1) + 1

    log.info(f'VGG-Conv output dims: H({hout}), W({wout}), C({channels})')
    return (nn.Sequential(*layers), hout * wout * channels)


def getGraphVgg(inputDims: List[int], pretrained=False, **kwargs):
    layers, outDim = makeVggLayers(VggCfgs['vggg'], inputDims)
    model = GraphVgg(layers, outDim, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(ModelUrls['vgg11']))

    return model


def getGraphVggBn(inputDims: List[int], pretrained=False, **kwargs):
    layers, outDim = makeVggLayers(VggCfgs['vggg'], inputDims, batchNorm=True)
    model = GraphVgg(layers, outDim, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(ModelUrls['vgg11_bn']))

    return model
