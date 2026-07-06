from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from einops import rearrange
from layers import *

try:
    import mmcv
    from mmcv.cnn import ResNet as MMCVResNet
    HAS_MMCV = True
except ImportError:
    HAS_MMCV = False
    MMCVResNet = None

if not HAS_MMCV:
    raise ImportError(
        "mmcv is not installed. This project requires mmcv to be installed. "
        "Please install mmcv by 'pip install mmcv' and retry. "
    )


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


class DepthEncoder(nn.Module):
    def __init__(self, num_layers, pretrained, num_input_images=1, use_mmcv=False):
        super(DepthEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        if use_mmcv:
            self.encoder = mmcv_resnet_multiimage_input(
                num_layers, pretrained, num_input_images
            )
        else:
            if num_input_images > 1:
                self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
            else:
                self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images=1)

    def forward(self, input_image):
        x = (input_image - 0.45) / 0.225
        features = self.encoder(x)
        return features


class PatchEmbedding(nn.Module):

    def __init__(self, in_channels=512, patch_size=1, emb_size=512, feature_size=(24, 80)):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.feature_size = feature_size
        h, w = feature_size
        self.num_patches = (h // patch_size) * (w // patch_size)

        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, emb_size))

    def forward(self, x):
        b, c, h, w = x.shape
        x = F.interpolate(x, size=self.feature_size, mode='bilinear', align_corners=True)

        x = self.proj(x)  
        x = rearrange(x, 'b c h w -> b (h w) c') 


        x = x + self.pos_embed

        return x


class TransformerBlock(nn.Module):

    def __init__(self, emb_size=512, num_heads=8, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.attn = nn.MultiheadAttention(emb_size, num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(emb_size)
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, emb_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_size * 4, emb_size),
            nn.Dropout(dropout)
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)
        x = residual + self.dropout1(attn_out)

        residual = x
        x = self.norm2(x)
        mlp_out = self.mlp(x)
        x = residual + self.dropout2(mlp_out)

        return x


class VisionTransformer(nn.Module):

    def __init__(self, in_channels=512, emb_size=512, num_layers=6, num_heads=8,
                 patch_size=1, feature_size=(24, 80), dropout=0.1):
        super(VisionTransformer, self).__init__()
        self.emb_size = emb_size
        self.feature_size = feature_size

        self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_size, feature_size)

        self.transformer_layers = nn.ModuleList([
            TransformerBlock(emb_size, num_heads, dropout) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(emb_size)

        self.reconstruct = nn.Linear(emb_size, emb_size * patch_size * patch_size)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.patch_embed(x)  
        for layer in self.transformer_layers:
            x = layer(x)

        x = self.norm(x)

        x = self.reconstruct(x)  
        patch_h, patch_w = self.feature_size[0] // self.patch_embed.patch_size, \
                           self.feature_size[1] // self.patch_embed.patch_size
        x = rearrange(x, 'b (h w) (c p1 p2) -> b c (h p1) (w p2)',
                      h=patch_h, w=patch_w,
                      p1=self.patch_embed.patch_size, p2=self.patch_embed.patch_size)

        return x


class DAMSAM(nn.Module):

    def __init__(self, in_channels, scales=[1, 2, 4, 8], reduction_ratio=8):
        super(DAMSAM, self).__init__()
        self.in_channels = in_channels
        self.scales = scales

        self.depth_enhance = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.GELU()
        )

        self.scale_branches = nn.ModuleList()
        for scale in scales:
            if scale == 1:

                branch = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels // reduction_ratio, 1),
                    nn.GELU(),
                    nn.Conv2d(in_channels // reduction_ratio, in_channels, 1)
                )
            else:
                branch = nn.Sequential(
                    nn.AvgPool2d(scale, stride=scale),
                    nn.Conv2d(in_channels, in_channels // reduction_ratio, 1),
                    nn.GELU(),
                    nn.Conv2d(in_channels // reduction_ratio, in_channels, 1),
                    nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)
                )
            self.scale_branches.append(branch)

        self.edge_aware = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 3, padding=1),
            nn.Sigmoid()
        )

        self.fusion_weights = nn.Parameter(torch.ones(len(scales)))

        self.gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.Sigmoid()
        )

        self.residual_enhance = nn.Conv2d(in_channels, in_channels, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = x

        enhanced = self.depth_enhance(x)

        scale_features = []
        for i, branch in enumerate(self.scale_branches):
            scale_feat = branch(enhanced)
            scale_features.append(scale_feat)

        weights = torch.softmax(self.fusion_weights, dim=0)
        fused = sum(w * f for w, f in zip(weights, scale_features))

        edge_attention = self.edge_aware(fused)

        attended = fused * edge_attention

        gate_input = torch.cat([attended, enhanced], dim=1)
        gate_weight = self.gate(gate_input)
        gated_output = gate_weight * attended + (1 - gate_weight) * enhanced

        output = gated_output + self.residual_enhance(residual)

        return output


class AdaptiveCrossScaleFusion(nn.Module):

    def __init__(self, high_channels, low_channels, output_channels=None):
        super(AdaptiveCrossScaleFusion, self).__init__()
        self.output_channels = output_channels or high_channels

        self.align = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.high_transform = nn.Sequential(
            nn.Conv2d(high_channels, self.output_channels // 2, 1),
            nn.GroupNorm(8, self.output_channels // 2),
            nn.GELU()
        )

        self.low_transform = nn.Sequential(
            nn.Conv2d(low_channels, self.output_channels // 2, 1),
            nn.GroupNorm(8, self.output_channels // 2),
            nn.GELU()
        )

        self.fusion_weight = nn.Sequential(
            nn.Conv2d(self.output_channels, self.output_channels // 4, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.output_channels // 4, 2, 1),
            nn.Softmax(dim=1)
        )

        self.detail_enhance = nn.Sequential(
            nn.Conv2d(self.output_channels, self.output_channels, 3, padding=1),
            nn.GroupNorm(8, self.output_channels),
            nn.GELU(),
            nn.Conv2d(self.output_channels, self.output_channels, 3, padding=1)
        )

    def forward(self, high_feat, low_feat):
        high_feat_aligned = self.align(high_feat)

        high_transformed = self.high_transform(high_feat_aligned)
        low_transformed = self.low_transform(low_feat)

        concat = torch.cat([high_transformed, low_transformed], dim=1)

        weights = self.fusion_weight(concat)  # [B, 2, H, W]
        weight_high = weights[:, 0:1, :, :]
        weight_low = weights[:, 1:2, :, :]

        fused = weight_high * high_transformed + weight_low * low_transformed

        output = fused + self.detail_enhance(fused)

        return output


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.vit = VisionTransformer(
            in_channels=512,
            emb_size=768,
            num_layers=12,
            num_heads=12, 
            patch_size=1, 
            feature_size=(6, 16) 
        )


        self.vit_adjust = nn.Sequential(
            nn.Conv2d(768, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.GELU()
        )

        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

            self.convs[("pixelshuffle", i)] = nn.Sequential(
                nn.Conv2d(self.num_ch_dec[i], self.num_ch_dec[i] * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(upscale_factor=2)
            )

            attention_in_channels = num_ch_in
            self.convs[("damsam", i)] = DAMSAM(
                attention_in_channels,
                scales=[1, 2, 4]
            )

            if self.use_skips and i > 0:
                self.convs[("cross_fusion", i)] = AdaptiveCrossScaleFusion(
                    high_channels=self.num_ch_dec[i],
                    low_channels=self.num_ch_enc[i - 1]
                )

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        x = input_features[-1]

        vit_out = self.vit(x)
        x = self.vit_adjust(vit_out)  

        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = self.convs[("pixelshuffle", i)](x)

            x = [x]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)

            x = self.convs[("damsam", i)](x)

            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs