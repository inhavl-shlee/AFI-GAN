# Copyright (c) Seongho Lee and SeungHwan Bae (InhaUniv.) All Rights Reserved.

import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F
import torch.nn as nn
import torch

from detectron2.layers import Conv2d, ShapeSpec, get_norm, ConvTranspose2d
from detectron2.utils.registry import Registry


import logging

logger = logging.getLogger(__name__)

class Discriminator(nn.Module):

    def  __init__(self):
        super(Discriminator, self).__init__()

        self.current_step = 0
        in_filters = 256
        self.kw = 3
        self.padw = 1
        self.stw = 1

        self.Discriminators = nn.ModuleList()
        first_discriminator = nn.ModuleList()

        f_mult = 1

        for n in range(1,4):
            f_mult_prev = f_mult
            f_mult = min(2**n,4)
            first_discriminator.append(
                nn.Sequential(Conv2d(in_filters * f_mult_prev, in_filters * f_mult, kernel_size=self.kw, stride=self.stw,
                                     padding=self.padw, norm=get_norm("BN", in_filters * f_mult)),
                              nn.LeakyReLU(0.2, True)))

        first_discriminator.append(nn.Sequential(
            Conv2d(in_filters * f_mult, 1, kernel_size=self.kw, stride=self.stw, padding=self.padw)))

        for sequential in first_discriminator:
            for layer in sequential:
                if not isinstance(layer,nn.LeakyReLU):
                    weight_init.c2_msra_fill(layer)

        first_discriminator = nn.Sequential(*first_discriminator)
        self.Discriminators.append(first_discriminator)

    def forward(self, feature):

        feature = self.Discriminators[self.current_step](feature)

        return feature
