from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

try:
    import mmcv
    from mmcv.cnn import ResNet as MMCVResNet
    HAS_MMCV = True
except ImportError:
    HAS_MMCV = False
    MMCVResNet = None


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
    assert num_layers in [18, 50], "Only ResNet18 and ResNet50 are supported."

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


class MMCVResNetWrapper(nn.Module):
    def __init__(self, mmcv_resnet):
        super().__init__()
        self.conv1 = mmcv_resnet.conv1
        self.maxpool = mmcv_resnet.maxpool
        self.res_layers = mmcv_resnet.res_layers
        self.layer1 = self.res_layers[0]
        self.layer2 = self.res_layers[1]
        self.layer3 = self.res_layers[2]
        self.layer4 = self.res_layers[3]

    def forward(self, x):
        x = self.conv1(x)
        features = [x]
        x = self.maxpool(x)
        for layer in self.res_layers:
            x = layer(x)
            features.append(x)
        return features


def mmcv_resnet_multiimage_input(num_layers, pretrained, num_input_images=1):
    if not HAS_MMCV:
        raise ImportError(
            "mmcv is not installed, cannot use MMCV backbone. "
            "Please install mmcv by 'pip install mmcv' and retry."
        )
    assert num_layers in [18, 50], "Only ResNet18 and ResNet50 are supported."

    init_cfg = None
    if pretrained:
        init_cfg = dict(
            type='Pretrained',
            checkpoint=f'torchvision://resnet{num_layers}'
        )

    model = MMCVResNet(
        depth=num_layers,
        in_channels=3,
        stem_channels=64,
        base_channels=64,
        num_stages=4,
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        out_indices=(4,),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        init_cfg=init_cfg
    )

    if pretrained:
        model.init_weights()

    if num_input_images > 1:
        old_conv = model.conv1.conv
        old_weight = old_conv.weight.data
        new_in_channels = num_input_images * 3
        new_weight = torch.cat([old_weight] * num_input_images, dim=1) / num_input_images

        new_conv = nn.Conv2d(
            new_in_channels, 64,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        new_conv.weight.data = new_weight
        model.conv1.conv = new_conv

    wrapper = MMCVResNetWrapper(model)
    return wrapper


class ResnetEncoderModify(nn.Module):
    def __init__(self, num_layers, pretrained, num_input_images=1, use_mmcv=True):
        super(ResnetEncoderModify, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        if use_mmcv:
            self.encoder = mmcv_resnet_multiimage_input(
                num_layers, pretrained, num_input_images
            )
        else:
            if num_input_images > 1:
                self.encoder = resnet_multiimage_input(
                    num_layers, pretrained, num_input_images
                )
            else:
                self.encoder = resnet_multiimage_input(
                    num_layers, pretrained, num_input_images=1
                )

    def forward(self, input_image):
        x = (input_image - 0.45) / 0.225
        features = self.encoder(x)
        return features


if __name__ == "__main__":
    encoder = ResnetEncoderModify(num_layers=18, pretrained=True, num_input_images=1)
    batch_size = 2
    test_input = torch.randn(batch_size, 3, 192, 640)
    features = encoder(test_input)
    print("Custom ResNet feature shapes:", [f.shape for f in features])

    if HAS_MMCV:
        encoder_mmcv = ResnetEncoderModify(
            num_layers=18, pretrained=True, num_input_images=1, use_mmcv=True
        )
        features_mmcv = encoder_mmcv(test_input)
        print("MMCV ResNet feature shapes:", [f.shape for f in features_mmcv])

        encoder_mmcv_multi = ResnetEncoderModify(
            num_layers=18, pretrained=True, num_input_images=2, use_mmcv=True
        )
        test_input_multi = torch.randn(batch_size, 6, 192, 640)
        features_multi = encoder_mmcv_multi(test_input_multi)
        print("MMCV multi-input feature shapes:", [f.shape for f in features_multi])
    else:
        print("mmcv is not installed, skipping MMCV tests.")