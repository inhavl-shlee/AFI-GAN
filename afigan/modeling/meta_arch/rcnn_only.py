# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
import torch
from torch import nn

from detectron2.structures import ImageList

from .build import GUIDE_ARCH_REGISTRY

from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY

__all__ = ["RCNN_FPN_only"]

@GUIDE_ARCH_REGISTRY.register()
class RCNN_FPN_only(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = self.build_backbone(cfg)
        self.input_format = cfg.INPUT.FORMAT

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs, img_dict_name='image'):


        images = [x[img_dict_name].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        features = self.backbone(images.tensor)
        processed_results = []
        processed_results.append({"features": features})
        return processed_results


    def build_backbone(self, cfg, input_shape=None):
        """
        Build a backbone from `cfg.MODEL.BACKBONE.NAME`.

        Returns:
            an instance of :class:`Backbone`
        """
        if input_shape is None:
            input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

        backbone_name = cfg.MODEL.GUIDE_BACKBONE.NAME
        backbone = BACKBONE_REGISTRY.get(backbone_name)(cfg, input_shape)
        assert isinstance(backbone, Backbone)
        return backbone