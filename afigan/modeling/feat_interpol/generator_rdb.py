# Copyright (c) Seongho Lee and SeungHwan Bae (InhaUniv.) All Rights Reserved.

import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F
import torch.nn as nn
import torch

from detectron2.layers import Conv2d, ShapeSpec, get_norm, ConvTranspose2d
from detectron2.utils.registry import Registry

import logging

logger = logging.getLogger(__name__)

class ResidualInResidual(nn.Module):
    def __init__(self, n_residual_dense_blocks ,in_features, growth_rate, residual_scale, kw, stw, padw):
        super(ResidualInResidual, self).__init__()

        RDBs = nn.ModuleList()

        for _ in range(n_residual_dense_blocks):
            RDBs.append(ResidualDenseBlock(in_features, growth_rate, residual_scale, kw, stw, padw))

        self.RDBs = nn.Sequential(*RDBs)
        self.residual_scale = residual_scale

    def forward(self, input):
        x = self.RDBs(input)

        return torch.add(x.mul(self.residual_scale),input)


class ResidualDenseBlock(nn.Module):
    def __init__(self, in_features, growth_rate, residual_scale, kw, stw, padw):
        super(ResidualDenseBlock, self).__init__()

        self.residual_scale = residual_scale

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features + 0 * growth_rate, growth_rate, kw, stw, padw, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_features + 1 * growth_rate, growth_rate, kw, stw, padw, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_features + 2 * growth_rate, growth_rate, kw, stw, padw, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_features + 3 * growth_rate, growth_rate, kw, stw, padw, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv5 = nn.Conv2d(in_features + 4 * growth_rate, in_features, kw, stw, padw, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(torch.cat((x, conv1),1))
        conv3 = self.conv3(torch.cat((x, conv1, conv2),1))
        conv4 = self.conv4(torch.cat((x, conv1, conv2, conv3),1))
        conv5 = self.conv5(torch.cat((x, conv1, conv2, conv3, conv4),1))

        return torch.add(x, conv5.mul(self.residual_scale))

class Generator(nn.Module):

    def __init__(self, in_channels=256, n_residual_dense_blocks=2, growth_rate = 32, residual_scale=0.2 , scale=2):
        super(Generator, self).__init__()

        self.in_channels = in_channels
        self.n_residual_dense_blocks = n_residual_dense_blocks
        self.growth_rate = growth_rate
        self.residual_scale = residual_scale
        self.scale = scale
        self.kw = 3
        self.padw = 1
        self.stw = 1

        self.Generators = nn.ModuleList()


        first_generator = nn.ModuleList()
        first_generator.append(nn.Sequential(
            Conv2d(self.in_channels, self.in_channels, kernel_size=self.kw, stride=self.stw, padding=self.padw),
            nn.LeakyReLU(0.2, True)))

        first_generator.append(ResidualInResidual(self.n_residual_dense_blocks, self.in_channels, self.growth_rate, self.residual_scale ,self.kw, self.stw, self.padw))

        first_generator.append(nn.Sequential(
            Conv2d(self.in_channels, self.in_channels, kernel_size=self.kw, stride=self.stw, padding=self.padw),
            nn.LeakyReLU(0.2, True)))

        first_generator.append(nn.Sequential(
            ConvTranspose2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=self.kw*2,
                           stride=self.stw*2, padding=self.padw*2, dilation=1),
            nn.LeakyReLU(0.2,True)
        ))

        first_generator.append(nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=self.kw, stride=self.stw, padding=self.padw)))

        for sequential in first_generator:
            if isinstance(sequential, ResidualInResidual) or isinstance(sequential, ResidualDenseBlock):
                continue
            for layer in sequential:
                if not isinstance(layer,nn.LeakyReLU) and not isinstance(layer,nn.PixelShuffle) and not isinstance(layer,nn.BatchNorm2d) and not isinstance(layer,nn.SyncBatchNorm):
                    nn.init.kaiming_normal_(layer.weight)
                    layer.weight.data *= 0.1
                    if layer.bias is not None:
                        layer.bias.data.zero_()

        first_generator = nn.Sequential(*first_generator)
        self.Generators.append(first_generator)

    def forward(self, features):

        inter_res = F.interpolate(features, scale_factor=2, mode="bilinear")

        features_gen = self.Generators[0](features)


        return torch.add(features_gen,inter_res)

