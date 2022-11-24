from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN


_C.MODEL.GUIDE_ARCHITECTURE = ""

_C.MODEL.GUIDE_WEIGHTS = ""
_C.MODEL.SRF_GEN_WEIGHTS = ""
_C.MODEL.SRF_DIS_WEIGHTS = ""
_C.MODEL.SRF_FREEZE = False

# ---------------------------------------------------------------------------- #
# Guide_Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.GUIDE_BACKBONE = CN()

_C.MODEL.GUIDE_BACKBONE.NAME = "build_resnet_fpn_backbone"
# Freeze the first several stages so they are not trained.
# There are 5 stages in ResNet. The first is a convolution, and the following
# stages are each group of residual blocks.
_C.MODEL.GUIDE_BACKBONE.FREEZE_AT = 2



# --------------------------------------------------------------------------- #
# For ResNeSt
# For more detail, please refer to https://github.com/chongruo/detectron2-ResNeSt
# --------------------------------------------------------------------------- #

# Radix in ResNeSt
_C.MODEL.RESNETS.RADIX = 1
# Bottleneck_width in ResNeSt
_C.MODEL.RESNETS.BOTTLENECK_WIDTH = 64
# Apply deep stem
_C.MODEL.RESNETS.DEEP_STEM = False
# Apply avg after conv2 in the BottleBlock
# When AVD=True, the STRIDE_IN_1X1 should be False
_C.MODEL.RESNETS.AVD = False
# Apply avg_down to the downsampling layer for residual path
_C.MODEL.RESNETS.AVG_DOWN = False


# ---------------------------------------------------------------------------- #
# BiFPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.BIFPN = CN()
# Names of the input feature maps to be used by FPN
# They must have contiguous power of 2 strides
# e.g., ["res2", "res3", "res4", "res5"]
_C.MODEL.BIFPN.IN_FEATURES = []
_C.MODEL.BIFPN.OUT_CHANNELS = 256
_C.MODEL.BIFPN.FPN_REPEAT = 3

# Options: "" (no norm), "GN"
_C.MODEL.BIFPN.NORM = "SyncBN"

# Types for fusing the FPN top-down and lateral features. Can be either "sum" or "avg"
_C.MODEL.BIFPN.FUSE_TYPE = "sum"

# ---------------------------------------------------------------------------- #
# Swin Transformer
# ---------------------------------------------------------------------------- #
# SwinT backbone
_C.MODEL.SWINT = CN()
_C.MODEL.SWINT.EMBED_DIM = 96
_C.MODEL.SWINT.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]
_C.MODEL.SWINT.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWINT.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWINT.WINDOW_SIZE = 7
_C.MODEL.SWINT.MLP_RATIO = 4
_C.MODEL.SWINT.DROP_PATH_RATE = 0.2
_C.MODEL.SWINT.APE = False
    # cfg.MODEL.BACKBONE.FREEZE_AT = -1
    # cfg.MODEL.FPN.TOP_LEVELS = 2
    # cfg.SOLVER.OPTIMIZER = "AdamW"

# Enable automatic mixed precision for training
# Note that this does not change model's inference behavior.
# To use AMP in inference, run inference under autocast()
_C.SOLVER.OPTIMIZER = "SGD"
_C.SOLVER.AMP = CN({"ENABLED": False})
# Gradient clipping
_C.SOLVER.CLIP_GRADIENTS = CN({"ENABLED": False})
# Type of gradient clipping, currently 2 values are supported:
# - "value": the absolute values of elements of each gradients are clipped
# - "norm": the norm of the gradient for each parameter is clipped thus
#   affecting all elements in the parameter
_C.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
# Maximum absolute value used for clipping gradients
_C.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
# Floating point number p for L-p norm to be used with the "norm"
# gradient clipping type; for L-inf, please specify .inf
_C.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0