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
from detectron2.modeling.backbone.resnet import build_resnet_backbone

from afigan.modeling.backbone.resnest import build_resnest_backbone
from afigan.modeling.feat_interpol import generator_rdb as G_rdb

__all__ = ["build_resnet_pafpn_sr_backbone", "build_resnest_pafpn_sr_backbone", "PAFPN_AFIGAN"]

class PAFPN_AFIGAN(Backbone):
    """
    This module implements Path Aggregation Feature Pyramid Network.
    """

    def __init__(
        self, bottom_up, in_features, out_channels, norm="", top_block=None, fuse_type="sum", cfg=None
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
        super(PAFPN_AFIGAN, self).__init__()
        assert isinstance(bottom_up, Backbone)
        self.cfg = cfg

        # Feature map strides and channels from the bottom up network (e.g. ResNet)
        input_shapes = bottom_up.output_shape()
        in_strides = [input_shapes[f].stride for f in in_features]
        in_channels = [input_shapes[f].channels for f in in_features]

        _assert_strides_are_log2_contiguous(in_strides)
        lateral_convs = []
        output_convs = []

        # PANET
        downsample_convs = []

        srf_module = G_rdb.Generator(n_residual_dense_blocks=3)

        if cfg.MODEL.SRF_FREEZE:
            for _idx, p in enumerate(srf_module.parameters()):
                p.requires_grad = False

        self.add_module("srf_module", srf_module)
        self.srf_module = srf_module

        use_bias = norm == ""
        for idx, in_channels in enumerate(in_channels):
            lateral_norm = get_norm(norm, out_channels)
            output_norm = get_norm(norm, out_channels)

            lateral_conv = Conv2d(
                in_channels, out_channels, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
            )

            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            stage = int(math.log2(in_strides[idx]))
            self.add_module("fpn_lateral{}".format(stage), lateral_conv)
            self.add_module("pafpn_output{}".format(stage), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)

            # PANET
            if idx > 0:
                downsample_norm = get_norm(norm, out_channels)
                downsample_conv = Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,  # for downsample
                    padding=1,
                    bias=use_bias,
                    norm=downsample_norm,
                )
                weight_init.c2_xavier_fill(downsample_conv)
                self.add_module("pafpn_downsample{}".format(stage), downsample_conv)
                downsample_convs.append(downsample_conv)

        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]  # 5 4 3 2

        # PANET
        self.output_convs = output_convs  # 2 3 4 5
        self.downsample_convs = downsample_convs  # 3 4 5
        self.top_block = top_block
        self.in_features = in_features
        self.bottom_up = bottom_up
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in in_strides}
        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = in_strides[-1]
        assert fuse_type in {"avg", "sum"}
        self._fuse_type = fuse_type




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
        bottom_up_features = self.bottom_up(x)
        x = [bottom_up_features[f] for f in self.in_features[::-1]]
        topdown_results = []
        results = []
        prev_features = self.lateral_convs[0](x[0])
        topdown_results.append(prev_features)
        # results.append(self.output_convs[0](prev_features))

        #TOP-DOWN PATHWAY
        for features, lateral_conv in zip(
            x[1:], self.lateral_convs[1:]
        ):
            # top_down_features = F.interpolate(prev_features, scale_factor=2, mode="nearest")
            top_down_features = self.srf_module(prev_features)
            lateral_features = lateral_conv(features)

            prev_features = lateral_features + top_down_features
            if self._fuse_type == "avg":
                prev_features /= 2
            topdown_results.insert(0, prev_features) # 2 3 4 5

        pa_prev_features = topdown_results.pop(0) # topdown : 3 4 5
        results.append(self.output_convs[0](pa_prev_features))

        #BOTTOM-UP AUGMENTATION
        for inter_feature, downsample_conv, output_conv in zip(topdown_results, self.downsample_convs, self.output_convs[1:]):
            downsampled_features = F.relu_(downsample_conv(pa_prev_features))

            pa_prev_features = inter_feature + downsampled_features
            if self._fuse_type == "avg":
                pa_prev_features /= 2
            results.append(output_conv(pa_prev_features))


        if self.top_block is not None:
            top_block_in_feature = bottom_up_features.get(self.top_block.in_feature, None)
            if top_block_in_feature is None:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        return dict(zip(self._out_features, results))

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


def _assert_strides_are_log2_contiguous(strides):
    """
    Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2".
    """
    for i, stride in enumerate(strides[1:], 1):
        assert stride == 2 * strides[i - 1], "Strides {} {} are not log2 contiguous".format(
            stride, strides[i - 1]
        )


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

@BACKBONE_REGISTRY.register()
def build_resnet_pafpn_sr_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = PAFPN_AFIGAN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
        cfg = cfg,
    )
    return backbone

@BACKBONE_REGISTRY.register()
def build_resnest_pafpn_sr_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnest_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = PAFPN_AFIGAN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
        cfg = cfg,
    )
    return backbone