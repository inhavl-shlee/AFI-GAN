# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.utils.registry import Registry


GUIDE_ARCH_REGISTRY = Registry("GUIDE_ARCH")
GUIDE_ARCH_REGISTRY.__doc__ = """ 

GUIDE_ARCH_REGISTRY : 
    PanopticFPN_only
    GeneralizedRCNN_only <NotImplemented>
"""


def build_guide_model(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    meta_guide_arch = cfg.MODEL.GUIDE_ARCHITECTURE
    return GUIDE_ARCH_REGISTRY.get(meta_guide_arch)(cfg)
