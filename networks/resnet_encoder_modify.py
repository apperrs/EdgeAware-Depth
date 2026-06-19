from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_input_images=1):
        super(ResNet, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self._init_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x) 

        x = self.maxpool(x)

        x = self.layer1(x)
        features.append(x) 

        x = self.layer2(x)
        features.append(x)  
        x = self.layer3(x)
        features.append(x) 

        x = self.layer4(x)
        features.append(x) 

        return features


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    assert num_layers in [18, 50], "只支持18层或50层的ResNet"

    if num_layers == 18:
        block = BasicBlock
        layers = [2, 2, 2, 2]
    else:  
        block = Bottleneck
        layers = [3, 4, 6, 3]

    model = ResNet(block, layers, num_input_images=num_input_images)

    if pretrained:
        model_urls = {
            'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
            'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
            'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
        }

        pretrained_dict = model_zoo.load_url(model_urls[f'resnet{num_layers}'])

        if num_input_images > 1:
            conv1_weight = pretrained_dict['conv1.weight']

            new_conv1_weight = torch.cat([conv1_weight] * num_input_images, 1) / num_input_images
            pretrained_dict['conv1.weight'] = new_conv1_weight

        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model


class ResnetEncoderModify(nn.Module):

    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoderModify, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images=1)

    def forward(self, input_image):
        x = (input_image - 0.45) / 0.225
        features = self.encoder(x)
        return features


if __name__ == "__main__":
    encoder = ResnetEncoderModify(num_layers=18, pretrained=True, num_input_images=1)

    batch_size = 2
    test_input = torch.randn(batch_size, 3, 192, 640)

    features = encoder(test_input)

    encoder_multi = ResnetEncoderModify(num_layers=18, pretrained=True, num_input_images=2)
    test_input_multi = torch.randn(batch_size, 6, 192, 640)
    features_multi = encoder_multi(test_input_multi)
