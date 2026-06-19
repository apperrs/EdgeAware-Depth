from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict

from einops import rearrange

from layers import *


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

        weights = self.fusion_weight(concat)  
        weight_high = weights[:, 0:1, :, :]
        weight_low = weights[:, 1:2, :, :]

        fused = weight_high * high_transformed + weight_low * low_transformed

        output = fused + self.detail_enhance(fused)

        return output


class DepthDecoderModify(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoderModify, self).__init__()

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
            feature_size=(6, 20)  
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
