# Baseline code : fpn.py # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by Seongho Lee and SeungHwan Bae (InhaUniv.), 2021. All Rights Reserved.

import math
import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F
import torch
from torch import nn

from detectron2.layers import Conv2d, ShapeSpec, get_norm

from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
# from detectron2.modeling.backbone.resnet import build_resnet_backbone

from afigan.modeling.bifpn_layers import Conv2d, SeparableConv2d, MaxPool2d, MemoryEfficientSwish, Swish
from afigan.modeling.backbone.swin_transformer import build_swint_backbone
from afigan.modeling.feat_interpol import generator_rdb as G_rdb

__all__ = ["build_swint_bifpn_sr_backbone", "BiFPN_AFIGAN"]

class Attention(nn.Module):
    def __init__(self, num_inputs, eps=1e-3):
        super().__init__()
        self.num_inputs = num_inputs
        self.atten_w = nn.Parameter(torch.ones(num_inputs))
        self.eps = eps

    def forward(self, inputs):
        assert isinstance(inputs, list) and len(inputs) == self.num_inputs
        atten_w = F.relu(self.atten_w)
        return sum(x_ * w for x_, w in zip(inputs, atten_w)) / (atten_w.sum() + self.eps)

    def __repr__(self):
        s = ('num_inputs={num_inputs}'
             ', eps={eps}')
        return self.__class__.__name__ +'('+ s.format(**self.__dict__) + ')'


class BiFPNLayer(nn.Module):
    def __init__(self, out_channels, in_channels=None, lateral=False, attention=True, norm='', epsilon=1e-4,
                 top_block=None, srf=None):
        super(BiFPNLayer, self).__init__()
        self.epsilon = epsilon
        self.attention = attention
        self.lateral = lateral
        self.srf = srf
        mom = 0.01
        eps = 1e-3

        # Conv layers
        self.conv6_up = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same', norm=norm,
                                        momentum=mom, eps=eps)
        self.conv5_up = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same', norm=norm,
                                        momentum=mom, eps=eps)
        self.conv4_up = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same', norm=norm,
                                        momentum=mom, eps=eps)
        self.conv3_up = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same', norm=norm,
                                        momentum=mom, eps=eps)
        self.conv4_down = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same', norm=norm,
                                          momentum=mom, eps=eps)
        self.conv5_down = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same', norm=norm,
                                          momentum=mom, eps=eps)
        self.conv6_down = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same', norm=norm,
                                          momentum=mom, eps=eps)
        self.conv7_down = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same', norm=norm,
                                          momentum=mom, eps=eps)
        if attention:
            self.init_attention_weights()
        self._downsample = MaxPool2d(3, 2)
        self._swish = MemoryEfficientSwish()

    def init_attention_weights(self):
        # top-down Weight
        self.p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        # down-top Weight
        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)

    def _weight_act(self, weight):
        weight = F.relu(weight)
        return weight / (torch.sum(weight, dim=0) + self.epsilon)

    def _attention(self, weight, inputs):
        assert isinstance(inputs, list) and len(inputs) == len(weight)
        return sum(x_ * w for x_, w in zip(inputs, weight))

    def _upsample(self, x):
        return self.srf(x)
        # return F.interpolate(x, scale_factor=2, mode='nearest')

    def _feature_funsion(self, cur_feature, top_feature, indice=-1):
        top_feature = self._upsample(top_feature)
        if self.attention and indice > 0:
            weight = getattr(self, "p{}_w1".format(indice))
            return self._attention(weight, [cur_feature, top_feature])
        else:
            return cur_feature + top_feature

    def _feature_funsion2(self, skip_feature, cur_feature, bottom_feature, indice=-1):
        _downsample = self._downsample(bottom_feature)
        if self.attention and indice > 0:
            if isinstance(skip_feature, torch.Tensor):
                weight = getattr(self, "p{}_w2".format(indice))
                return self._attention(weight, [skip_feature, cur_feature, _downsample])
            else:
                weight = getattr(self, "p{}_w2".format(indice))
                return self._attention(weight, [cur_feature, _downsample])
        else:
            if isinstance(skip_feature, torch.Tensor):
                return skip_feature + cur_feature + _downsample
            else:
                return cur_feature + _downsample

    def _forward_up(self, inputs):
        p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        # print( p3_in.shape, p4_in.shape, p5_in.shape, p6_in.shape, p7_in.shape)
        # Connections for P6_0 and P7_0 to P6_1 respectively
        p6_up = self.conv6_up(self._swish(self._feature_funsion(p6_in, p7_in, 6)))
        p5_up = self.conv5_up(self._swish(self._feature_funsion(p5_in, p6_up, 5)))
        p4_up = self.conv4_up(self._swish(self._feature_funsion(p4_in, p5_up, 4)))
        p3_up = self.conv3_up(self._swish(self._feature_funsion(p3_in, p4_up, 3)))
        return p3_up, p4_up, p5_up, p6_up, p7_in

    def _forward_down(self, laterals, up_features, skip_features):
        _, p4_in, p5_in, p6_in, p7_in = laterals
        p3_up, p4_up, p5_up, p6_up, _ = up_features

        if self.lateral:
            p4_in, p5_in = skip_features

        p4_out = self.conv4_down(self._swish(
            self._feature_funsion2(p4_in, p4_up, p3_up, 4)))
        p5_out = self.conv5_down(self._swish(
            self._feature_funsion2(p5_in, p5_up, p4_out, 5)))
        p6_out = self.conv6_down(self._swish(
            self._feature_funsion2(p6_in, p6_up, p5_out, 6)))
        p7_out = self.conv7_down(self._swish(
            self._feature_funsion2(None, p7_in, p6_out, 7)))
        return p3_up, p4_out, p5_out, p6_out, p7_out

    def forward(self, inputs):
        laterals, skip_features = inputs
        up_features = self._forward_up(laterals)

        return self._forward_down(laterals, up_features, skip_features), None

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class BeforeBiFPNLayer(nn.Module):
    def __init__(self, out_channels, in_channels=None, epsilon=1e-4, top_block=None):
        super(BeforeBiFPNLayer, self).__init__()
        self.epsilon = epsilon
        mom = 0.01
        eps = 1e-3

        self.lateral3 = nn.Sequential(
            Conv2d(in_channels[0], out_channels, 1, stride=1, padding_mode='static_same'),
            nn.BatchNorm2d(out_channels, momentum=mom, eps=eps),
        )
        self.lateral4 = nn.Sequential(
            Conv2d(in_channels[1], out_channels, 1, stride=1, padding_mode='static_same'),
            nn.BatchNorm2d(out_channels, momentum=mom, eps=eps),
        )
        self.lateral5 = nn.Sequential(
            Conv2d(in_channels[2], out_channels, 1, stride=1, padding_mode='static_same'),
            nn.BatchNorm2d(out_channels, momentum=mom, eps=eps),
        )
        self.top_block = top_block

        # backbone skip connection
        self.p4_skip = nn.Sequential(
            Conv2d(in_channels[1], out_channels, 1, stride=1, padding_mode='static_same'),
            nn.BatchNorm2d(out_channels, momentum=mom, eps=eps),
        )
        self.p5_skip = nn.Sequential(
            Conv2d(in_channels[2], out_channels, 1, stride=1, padding_mode='static_same'),
            nn.BatchNorm2d(out_channels, momentum=mom, eps=eps),
        )

    def forward(self, inputs):
        c3, c4, c5 = inputs

        c4_skip = self.p4_skip(c4)
        c5_skip = self.p5_skip(c5)
        c6, c7 = self.top_block(c5)

        c3 = self.lateral3(c3)
        c4 = self.lateral4(c4)
        c5 = self.lateral5(c5)

        return (c3, c4, c5, c6, c7), (c4_skip, c5_skip)

class BiFPN_AFIGAN(Backbone):
    """
    This module implements Bi-directional Feature Pyramid Network.
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(
        self, bottom_up, in_features, out_channels, fpn_repeat, norm="SyncBN", top_block=None, fuse_type="sum", cfg=None
    ):
        """
        Args:
            bottom_up (Backbone): module representing the bottom up subnetwork.
                Must be a subclass of :class:`Backbone`. The multi-scale feature
                maps generated by the bottom up network, and listed in `in_features`,
                are used to generate FPN levels.
            in_features (list[str]): names of the input feature maps coming
                from the backbone to which FPN is attached. For example, if the
                backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
                of these may be used; order must be from high to low resolution.
            out_channels (int): number of channels in the output feature maps.
            norm (str): the normalization to use.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra FPN levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            fuse_type (str): types for fusing the top down features and the lateral
                ones. It can be "sum" (default), which sums up element-wise; or "avg",
                which takes the element-wise mean of the two.
        """
        #TODO: fpn_repeat (int): number of repeated block for building biFPN layers

        super(BiFPN_AFIGAN, self).__init__()
        assert isinstance(bottom_up, Backbone)

        # Feature map strides and channels from the bottom up network (e.g. ResNet)

        in_strides = [bottom_up._out_feature_strides[f] for f in in_features]
        in_channels = [bottom_up._out_feature_channels[f] for f in in_features]

        self.in_features = in_features
        self.bottom_up = bottom_up

        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {f"p{int(math.log2(s))}": s for s in in_strides}

        self.cfg = cfg


        # top block output feature maps.
        if top_block is not None:
            last_stage = int(math.log2(in_strides[-1]))
        for s in range(last_stage, last_stage + top_block.num_levels):
            in_strides.append(2 ** (s + 1))
            in_channels.append(out_channels)
            self._out_feature_strides[f"p{s + 1}"] = 2 ** (s + 1)

        _assert_strides_are_log2_contiguous(in_strides)

        self.before_bifpn = BeforeBiFPNLayer(out_channels, in_channels, top_block=top_block)
        layers = []
        lateral = True
        first_layer = True

        # Common layers
        srf_module = G_rdb.Generator(n_residual_dense_blocks=3)
        if cfg.MODEL.SRF_FREEZE:
            for _idx, p in enumerate(srf_module.parameters()):
                p.requires_grad = False

        self.add_module("srf_module", srf_module)
        self.srf_module = srf_module
        self._downsample = MaxPool2d(3, 2)
        self._swish = MemoryEfficientSwish()
        # parameters
        mom = 0.01
        eps = 1e-3
        self.attention = True
        # BiFPNLayer_0
        # Conv layers
        self.BiFPNLayer_0_conv6_up = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same', norm=norm,
                                        momentum=mom, eps=eps)
        self.BiFPNLayer_0_conv5_up = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same', norm=norm,
                                        momentum=mom, eps=eps)
        self.BiFPNLayer_0_conv4_up = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same', norm=norm,
                                        momentum=mom, eps=eps)
        self.BiFPNLayer_0_conv3_up = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same', norm=norm,
                                        momentum=mom, eps=eps)
        self.BiFPNLayer_0_conv4_down = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same', norm=norm,
                                          momentum=mom, eps=eps)
        self.BiFPNLayer_0_conv5_down = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same', norm=norm,
                                          momentum=mom, eps=eps)
        self.BiFPNLayer_0_conv6_down = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same', norm=norm,
                                          momentum=mom, eps=eps)
        self.BiFPNLayer_0_conv7_down = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same', norm=norm,
                                          momentum=mom, eps=eps)
        # top-down Weight
        self.BiFPNLayer_0_p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.BiFPNLayer_0_p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.BiFPNLayer_0_p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.BiFPNLayer_0_p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        # down-top Weight
        self.BiFPNLayer_0_p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.BiFPNLayer_0_p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.BiFPNLayer_0_p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.BiFPNLayer_0_p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)

        # BiFPNLayer_1
        # Conv layers
        self.BiFPNLayer_1_conv6_up = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same', norm=norm,
                                        momentum=mom, eps=eps)
        self.BiFPNLayer_1_conv5_up = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same', norm=norm,
                                        momentum=mom, eps=eps)
        self.BiFPNLayer_1_conv4_up = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same', norm=norm,
                                        momentum=mom, eps=eps)
        self.BiFPNLayer_1_conv3_up = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same', norm=norm,
                                        momentum=mom, eps=eps)
        self.BiFPNLayer_1_conv4_down = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same', norm=norm,
                                          momentum=mom, eps=eps)
        self.BiFPNLayer_1_conv5_down = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same', norm=norm,
                                          momentum=mom, eps=eps)
        self.BiFPNLayer_1_conv6_down = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same', norm=norm,
                                          momentum=mom, eps=eps)
        self.BiFPNLayer_1_conv7_down = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same', norm=norm,
                                          momentum=mom, eps=eps)
        # top-down Weight
        self.BiFPNLayer_1_p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.BiFPNLayer_1_p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.BiFPNLayer_1_p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.BiFPNLayer_1_p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        # down-top Weight
        self.BiFPNLayer_1_p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.BiFPNLayer_1_p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.BiFPNLayer_1_p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.BiFPNLayer_1_p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)

        # BiFPNLayer_2
        # Conv layers
        self.BiFPNLayer_2_conv6_up = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same', norm=norm,
                                        momentum=mom, eps=eps)
        self.BiFPNLayer_2_conv5_up = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same', norm=norm,
                                        momentum=mom, eps=eps)
        self.BiFPNLayer_2_conv4_up = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same', norm=norm,
                                        momentum=mom, eps=eps)
        self.BiFPNLayer_2_conv3_up = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same', norm=norm,
                                        momentum=mom, eps=eps)
        self.BiFPNLayer_2_conv4_down = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same', norm=norm,
                                          momentum=mom, eps=eps)
        self.BiFPNLayer_2_conv5_down = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same', norm=norm,
                                          momentum=mom, eps=eps)
        self.BiFPNLayer_2_conv6_down = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same', norm=norm,
                                          momentum=mom, eps=eps)
        self.BiFPNLayer_2_conv7_down = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same', norm=norm,
                                          momentum=mom, eps=eps)
        # top-down Weight
        self.BiFPNLayer_2_p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.BiFPNLayer_2_p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.BiFPNLayer_2_p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.BiFPNLayer_2_p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        # down-top Weight
        self.BiFPNLayer_2_p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.BiFPNLayer_2_p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.BiFPNLayer_2_p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.BiFPNLayer_2_p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)

        ########CUSTOM_LAYER#################
        # BiFPNLayer_3
        # Conv layers
        self.BiFPNLayer_3_conv6_up = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same',
                                                     norm=norm,
                                                     momentum=mom, eps=eps)
        self.BiFPNLayer_3_conv5_up = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same',
                                                     norm=norm,
                                                     momentum=mom, eps=eps)
        self.BiFPNLayer_3_conv4_up = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same',
                                                     norm=norm,
                                                     momentum=mom, eps=eps)
        self.BiFPNLayer_3_conv3_up = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same',
                                                     norm=norm,
                                                     momentum=mom, eps=eps)
        self.BiFPNLayer_3_conv4_down = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same',
                                                       norm=norm,
                                                       momentum=mom, eps=eps)
        self.BiFPNLayer_3_conv5_down = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same',
                                                       norm=norm,
                                                       momentum=mom, eps=eps)
        self.BiFPNLayer_3_conv6_down = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same',
                                                       norm=norm,
                                                       momentum=mom, eps=eps)
        self.BiFPNLayer_3_conv7_down = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same',
                                                       norm=norm,
                                                       momentum=mom, eps=eps)
        # top-down Weight
        self.BiFPNLayer_3_p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.BiFPNLayer_3_p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.BiFPNLayer_3_p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.BiFPNLayer_3_p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        # down-top Weight
        self.BiFPNLayer_3_p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.BiFPNLayer_3_p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.BiFPNLayer_3_p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.BiFPNLayer_3_p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)

        # BiFPNLayer_4
        # Conv layers
        self.BiFPNLayer_4_conv6_up = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same',
                                                     norm=norm,
                                                     momentum=mom, eps=eps)
        self.BiFPNLayer_4_conv5_up = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same',
                                                     norm=norm,
                                                     momentum=mom, eps=eps)
        self.BiFPNLayer_4_conv4_up = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same',
                                                     norm=norm,
                                                     momentum=mom, eps=eps)
        self.BiFPNLayer_4_conv3_up = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same',
                                                     norm=norm,
                                                     momentum=mom, eps=eps)
        self.BiFPNLayer_4_conv4_down = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same',
                                                       norm=norm,
                                                       momentum=mom, eps=eps)
        self.BiFPNLayer_4_conv5_down = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same',
                                                       norm=norm,
                                                       momentum=mom, eps=eps)
        self.BiFPNLayer_4_conv6_down = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same',
                                                       norm=norm,
                                                       momentum=mom, eps=eps)
        self.BiFPNLayer_4_conv7_down = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same',
                                                       norm=norm,
                                                       momentum=mom, eps=eps)
        # top-down Weight
        self.BiFPNLayer_4_p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.BiFPNLayer_4_p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.BiFPNLayer_4_p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.BiFPNLayer_4_p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        # down-top Weight
        self.BiFPNLayer_4_p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.BiFPNLayer_4_p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.BiFPNLayer_4_p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.BiFPNLayer_4_p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)

        # BiFPNLayer_5
        # Conv layers
        self.BiFPNLayer_5_conv6_up = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same',
                                                     norm=norm,
                                                     momentum=mom, eps=eps)
        self.BiFPNLayer_5_conv5_up = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same',
                                                     norm=norm,
                                                     momentum=mom, eps=eps)
        self.BiFPNLayer_5_conv4_up = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same',
                                                     norm=norm,
                                                     momentum=mom, eps=eps)
        self.BiFPNLayer_5_conv3_up = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same',
                                                     norm=norm,
                                                     momentum=mom, eps=eps)
        self.BiFPNLayer_5_conv4_down = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same',
                                                       norm=norm,
                                                       momentum=mom, eps=eps)
        self.BiFPNLayer_5_conv5_down = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same',
                                                       norm=norm,
                                                       momentum=mom, eps=eps)
        self.BiFPNLayer_5_conv6_down = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same',
                                                       norm=norm,
                                                       momentum=mom, eps=eps)
        self.BiFPNLayer_5_conv7_down = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same',
                                                       norm=norm,
                                                       momentum=mom, eps=eps)
        # top-down Weight
        self.BiFPNLayer_5_p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.BiFPNLayer_5_p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.BiFPNLayer_5_p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.BiFPNLayer_5_p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        # down-top Weight
        self.BiFPNLayer_5_p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.BiFPNLayer_5_p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.BiFPNLayer_5_p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.BiFPNLayer_5_p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)

        # BiFPNLayer_6
        # Conv layers
        self.BiFPNLayer_6_conv6_up = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same',
                                                     norm=norm,
                                                     momentum=mom, eps=eps)
        self.BiFPNLayer_6_conv5_up = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same',
                                                     norm=norm,
                                                     momentum=mom, eps=eps)
        self.BiFPNLayer_6_conv4_up = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same',
                                                     norm=norm,
                                                     momentum=mom, eps=eps)
        self.BiFPNLayer_6_conv3_up = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same',
                                                     norm=norm,
                                                     momentum=mom, eps=eps)
        self.BiFPNLayer_6_conv4_down = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same',
                                                       norm=norm,
                                                       momentum=mom, eps=eps)
        self.BiFPNLayer_6_conv5_down = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same',
                                                       norm=norm,
                                                       momentum=mom, eps=eps)
        self.BiFPNLayer_6_conv6_down = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same',
                                                       norm=norm,
                                                       momentum=mom, eps=eps)
        self.BiFPNLayer_6_conv7_down = SeparableConv2d(out_channels, out_channels, 3, padding_mode='static_same',
                                                       norm=norm,
                                                       momentum=mom, eps=eps)
        # top-down Weight
        self.BiFPNLayer_6_p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.BiFPNLayer_6_p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.BiFPNLayer_6_p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.BiFPNLayer_6_p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        # down-top Weight
        self.BiFPNLayer_6_p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.BiFPNLayer_6_p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.BiFPNLayer_6_p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.BiFPNLayer_6_p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)

        #######################################################################################################


        self.bifpn = nn.Sequential(*layers)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = self._out_feature_strides[self._out_features[-1]]

        assert fuse_type in {"avg", "sum"}
        self._fuse_type = fuse_type

    def _weight_act(self, weight):
        weight = F.relu(weight)
        return weight / (torch.sum(weight, dim=0) + self.epsilon)

    def _attention(self, weight, inputs):
        assert isinstance(inputs, list) and len(inputs) == len(weight)
        return sum(x_ * w for x_, w in zip(inputs, weight))

    def _upsample(self, x):
        return self.srf_module(x)

    def _feature_funsion(self, layer_idx,cur_feature, top_feature, indice=-1):
        top_feature = self._upsample(top_feature)
        if self.attention and indice > 0:
            weight = getattr(self, "BiFPNLayer_{}_p{}_w1".format(layer_idx,indice))
            return self._attention(weight, [cur_feature, top_feature])
        else:
            return cur_feature + top_feature

    def _feature_funsion2(self, layer_idx, skip_feature, cur_feature, bottom_feature, indice=-1):
        _downsample = self._downsample(bottom_feature)
        if self.attention and indice > 0:
            if isinstance(skip_feature, torch.Tensor):
                weight = getattr(self, "BiFPNLayer_{}_p{}_w2".format(layer_idx,indice))
                return self._attention(weight, [skip_feature, cur_feature, _downsample])
            else:
                weight = getattr(self, "BiFPNLayer_{}_p{}_w2".format(layer_idx,indice))
                return self._attention(weight, [cur_feature, _downsample])
        else:
            if isinstance(skip_feature, torch.Tensor):
                return skip_feature + cur_feature + _downsample
            else:
                return cur_feature + _downsample

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        # Reverse feature maps into top-down order (from low to high resolution)
        layer_idx = 0
        bottom_up_features = self.bottom_up(x)
        features = [bottom_up_features[f] for f in self.in_features]

        lateral_features, skip_features = self.before_bifpn(features)

        laterals, skip_features = lateral_features, skip_features
        p3_in, p4_in, p5_in, p6_in, p7_in = laterals
        p6_up = self.BiFPNLayer_0_conv6_up(self._swish(self._feature_funsion(0,p6_in, p7_in, 6)))
        p5_up = self.BiFPNLayer_0_conv5_up(self._swish(self._feature_funsion(0,p5_in, p6_up, 5)))
        p4_up = self.BiFPNLayer_0_conv4_up(self._swish(self._feature_funsion(0,p4_in, p5_up, 4)))
        p3_up = self.BiFPNLayer_0_conv3_up(self._swish(self._feature_funsion(0,p3_in, p4_up, 3)))

        _, p4_in, p5_in, p6_in, p7_in = lateral_features

        p4_in, p5_in = skip_features

        p4_out = self.BiFPNLayer_0_conv4_down(self._swish(
            self._feature_funsion2(0,p4_in, p4_up, p3_up, 4)))
        p5_out = self.BiFPNLayer_0_conv5_down(self._swish(
            self._feature_funsion2(0,p5_in, p5_up, p4_out, 5)))
        p6_out = self.BiFPNLayer_0_conv6_down(self._swish(
            self._feature_funsion2(0,p6_in, p6_up, p5_out, 6)))
        p7_out = self.BiFPNLayer_0_conv7_down(self._swish(
            self._feature_funsion2(0, None, p7_in, p6_out, 7)))
        features = (p3_up, p4_out, p5_out, p6_out, p7_out)


        laterals = features
        p3_in, p4_in, p5_in, p6_in, p7_in = laterals
        p6_up = self.BiFPNLayer_1_conv6_up(self._swish(self._feature_funsion(1,p6_in, p7_in, 6)))
        p5_up = self.BiFPNLayer_1_conv5_up(self._swish(self._feature_funsion(1,p5_in, p6_up, 5)))
        p4_up = self.BiFPNLayer_1_conv4_up(self._swish(self._feature_funsion(1,p4_in, p5_up, 4)))
        p3_up = self.BiFPNLayer_1_conv3_up(self._swish(self._feature_funsion(1,p3_in, p4_up, 3)))

        _, p4_in, p5_in, p6_in, p7_in = lateral_features
        p4_out = self.BiFPNLayer_1_conv4_down(self._swish(
            self._feature_funsion2(1,p4_in, p4_up, p3_up, 4)))
        p5_out = self.BiFPNLayer_1_conv5_down(self._swish(
            self._feature_funsion2(1,p5_in, p5_up, p4_out, 5)))
        p6_out = self.BiFPNLayer_1_conv6_down(self._swish(
            self._feature_funsion2(1,p6_in, p6_up, p5_out, 6)))
        p7_out = self.BiFPNLayer_1_conv7_down(self._swish(
            self._feature_funsion2(1,None, p7_in, p6_out, 7)))
        features = (p3_up, p4_out, p5_out, p6_out, p7_out)

        laterals = features
        p3_in, p4_in, p5_in, p6_in, p7_in = laterals
        p6_up = self.BiFPNLayer_2_conv6_up(self._swish(self._feature_funsion(2,p6_in, p7_in, 6)))
        p5_up = self.BiFPNLayer_2_conv5_up(self._swish(self._feature_funsion(2,p5_in, p6_up, 5)))
        p4_up = self.BiFPNLayer_2_conv4_up(self._swish(self._feature_funsion(2,p4_in, p5_up, 4)))
        p3_up = self.BiFPNLayer_2_conv3_up(self._swish(self._feature_funsion(2,p3_in, p4_up, 3)))

        _, p4_in, p5_in, p6_in, p7_in = lateral_features



        p4_out = self.BiFPNLayer_2_conv4_down(self._swish(
            self._feature_funsion2(2,p4_in, p4_up, p3_up, 4)))
        p5_out = self.BiFPNLayer_2_conv5_down(self._swish(
            self._feature_funsion2(2,p5_in, p5_up, p4_out, 5)))
        p6_out = self.BiFPNLayer_2_conv6_down(self._swish(
            self._feature_funsion2(2,p6_in, p6_up, p5_out, 6)))
        p7_out = self.BiFPNLayer_2_conv7_down(self._swish(
            self._feature_funsion2(2, None, p7_in, p6_out, 7)))
        features = (p3_up, p4_out, p5_out, p6_out, p7_out)

        ######################################
        # 3rd iter
        laterals = features
        p3_in, p4_in, p5_in, p6_in, p7_in = laterals
        p6_up = self.BiFPNLayer_3_conv6_up(self._swish(self._feature_funsion(3, p6_in, p7_in, 6)))
        p5_up = self.BiFPNLayer_3_conv5_up(self._swish(self._feature_funsion(3, p5_in, p6_up, 5)))
        p4_up = self.BiFPNLayer_3_conv4_up(self._swish(self._feature_funsion(3, p4_in, p5_up, 4)))
        p3_up = self.BiFPNLayer_3_conv3_up(self._swish(self._feature_funsion(3, p3_in, p4_up, 3)))

        _, p4_in, p5_in, p6_in, p7_in = lateral_features

        p4_out = self.BiFPNLayer_3_conv4_down(self._swish(
            self._feature_funsion2(3, p4_in, p4_up, p3_up, 4)))
        p5_out = self.BiFPNLayer_3_conv5_down(self._swish(
            self._feature_funsion2(3, p5_in, p5_up, p4_out, 5)))
        p6_out = self.BiFPNLayer_3_conv6_down(self._swish(
            self._feature_funsion2(3, p6_in, p6_up, p5_out, 6)))
        p7_out = self.BiFPNLayer_3_conv7_down(self._swish(
            self._feature_funsion2(3, None, p7_in, p6_out, 7)))
        features = (p3_up, p4_out, p5_out, p6_out, p7_out)

        # 4th iter
        laterals = features
        p3_in, p4_in, p5_in, p6_in, p7_in = laterals
        p6_up = self.BiFPNLayer_4_conv6_up(self._swish(self._feature_funsion(4, p6_in, p7_in, 6)))
        p5_up = self.BiFPNLayer_4_conv5_up(self._swish(self._feature_funsion(4, p5_in, p6_up, 5)))
        p4_up = self.BiFPNLayer_4_conv4_up(self._swish(self._feature_funsion(4, p4_in, p5_up, 4)))
        p3_up = self.BiFPNLayer_4_conv3_up(self._swish(self._feature_funsion(4, p3_in, p4_up, 3)))

        _, p4_in, p5_in, p6_in, p7_in = lateral_features

        p4_out = self.BiFPNLayer_4_conv4_down(self._swish(
            self._feature_funsion2(4, p4_in, p4_up, p3_up, 4)))
        p5_out = self.BiFPNLayer_4_conv5_down(self._swish(
            self._feature_funsion2(4, p5_in, p5_up, p4_out, 5)))
        p6_out = self.BiFPNLayer_4_conv6_down(self._swish(
            self._feature_funsion2(4, p6_in, p6_up, p5_out, 6)))
        p7_out = self.BiFPNLayer_4_conv7_down(self._swish(
            self._feature_funsion2(4, None, p7_in, p6_out, 7)))
        features = (p3_up, p4_out, p5_out, p6_out, p7_out)

        # 5th iter
        laterals = features
        p3_in, p4_in, p5_in, p6_in, p7_in = laterals
        p6_up = self.BiFPNLayer_5_conv6_up(self._swish(self._feature_funsion(5, p6_in, p7_in, 6)))
        p5_up = self.BiFPNLayer_5_conv5_up(self._swish(self._feature_funsion(5, p5_in, p6_up, 5)))
        p4_up = self.BiFPNLayer_5_conv4_up(self._swish(self._feature_funsion(5, p4_in, p5_up, 4)))
        p3_up = self.BiFPNLayer_5_conv3_up(self._swish(self._feature_funsion(5, p3_in, p4_up, 3)))

        _, p4_in, p5_in, p6_in, p7_in = lateral_features

        p4_out = self.BiFPNLayer_5_conv4_down(self._swish(
            self._feature_funsion2(5, p4_in, p4_up, p3_up, 4)))
        p5_out = self.BiFPNLayer_5_conv5_down(self._swish(
            self._feature_funsion2(5, p5_in, p5_up, p4_out, 5)))
        p6_out = self.BiFPNLayer_5_conv6_down(self._swish(
            self._feature_funsion2(5, p6_in, p6_up, p5_out, 6)))
        p7_out = self.BiFPNLayer_5_conv7_down(self._swish(
            self._feature_funsion2(5, None, p7_in, p6_out, 7)))
        features = (p3_up, p4_out, p5_out, p6_out, p7_out)

        # 6th iter
        laterals = features
        p3_in, p4_in, p5_in, p6_in, p7_in = laterals
        p6_up = self.BiFPNLayer_6_conv6_up(self._swish(self._feature_funsion(6, p6_in, p7_in, 6)))
        p5_up = self.BiFPNLayer_6_conv5_up(self._swish(self._feature_funsion(6, p5_in, p6_up, 5)))
        p4_up = self.BiFPNLayer_6_conv4_up(self._swish(self._feature_funsion(6, p4_in, p5_up, 4)))
        p3_up = self.BiFPNLayer_6_conv3_up(self._swish(self._feature_funsion(6, p3_in, p4_up, 3)))

        _, p4_in, p5_in, p6_in, p7_in = lateral_features

        p4_out = self.BiFPNLayer_6_conv4_down(self._swish(
            self._feature_funsion2(6, p4_in, p4_up, p3_up, 4)))
        p5_out = self.BiFPNLayer_6_conv5_down(self._swish(
            self._feature_funsion2(6, p5_in, p5_up, p4_out, 5)))
        p6_out = self.BiFPNLayer_6_conv6_down(self._swish(
            self._feature_funsion2(6, p6_in, p6_up, p5_out, 6)))
        p7_out = self.BiFPNLayer_6_conv7_down(self._swish(
            self._feature_funsion2(6, None, p7_in, p6_out, 7)))
        features = (p3_up, p4_out, p5_out, p6_out, p7_out)
        ######################################

        assert len(self._out_features) == len(features)
        return dict(zip(self._out_features, features))


def _assert_strides_are_log2_contiguous(strides):
    """
    Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2".
    """
    for i, stride in enumerate(strides[1:], 1):
        assert stride == 2 * strides[i - 1], "Strides {} {} are not log2 contiguous".format(
            stride, strides[i - 1]
        )

class ResampleFeature(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, norm):
        super().__init__()
        self.conv = Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding_mode="static_same")
        self.norm = get_norm(norm, out_channels) if norm != '' else lambda x: x
        self.resample = MaxPool2d(kernel_size=3, stride=2, padding_mode="static_same")

        weight_init.c2_xavier_fill(self.conv)

    def forward(self, x):
        return self.resample(self.norm(self.conv(x)))


class LastLevelMaxPool(nn.Module):
    """
    This module is used in the original FPN to generate a downsampled
    P6 feature from P5.
    """

    def __init__(self):
        super().__init__()
        self.num_levels = 1
        self.in_feature = "p5"

    def forward(self, x):
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]


class LastLevelP6P7(nn.Module):
    """
      This module is used in RetinaNet to generate extra layers, P6 and P7 from
      C5 feature.
    """
    def __init__(self, in_channels, out_channels, norm=''):
        super().__init__()
        self.num_levels = 2
        self.p6 = ResampleFeature(in_channels, out_channels, 1, norm=norm)
        self.p7 = MaxPool2d(kernel_size=3, stride=2, padding_mode="static_same")
        # ResampleFeature(out_channels, out_channels, 1, norm=norm)

    def forward(self, p5):
        p6 = self.p6(p5)
        p7 = self.p7(p6)
        return [p6, p7]


@BACKBONE_REGISTRY.register()
def build_swint_bifpn_sr_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_swint_backbone(cfg, input_shape)
    in_features = cfg.MODEL.BIFPN.IN_FEATURES
    out_channels = cfg.MODEL.BIFPN.OUT_CHANNELS
    in_channels_p6p7 = bottom_up.output_shape()[in_features[-1]].channels
    backbone = BiFPN_AFIGAN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        fpn_repeat=cfg.MODEL.BIFPN.FPN_REPEAT,
        norm=cfg.MODEL.BIFPN.NORM,
        # top_block=LastLevelMaxPool(),
        top_block=LastLevelP6P7(in_channels_p6p7,
                                out_channels,
                                cfg.MODEL.BIFPN.NORM),
        fuse_type=cfg.MODEL.BIFPN.FUSE_TYPE,
        cfg=cfg,
    )
    return backbone

