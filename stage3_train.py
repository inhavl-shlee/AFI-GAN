# Baseline code : train_net.py from detectron2 #
# stage1: AFI-GAN training for learning a generalized AFI-GAN

import logging
import os
from collections import OrderedDict
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.engine import default_argument_parser, default_setup, hooks, launch
from afigan.engine import Target_Detector_Trainer
from detectron2.evaluation import (
    DatasetEvaluators,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA

from afigan.evaluation import (
    COCOEvaluator,
)
from afigan.config import get_cfg

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    """
    Evaluation function
    :param args:
    :return: AFI-GAN training
    """
    cfg = setup(args)

    trainer = Target_Detector_Trainer(cfg)

    if args.resume:
        trainer.resume_or_load(resume=args.resume)
    else:
        trainer.resume_or_load(resume=args.resume)
        trainer.load_AFExtractor_weight(cfg.MODEL.AF_EXTRACTOR_WEIGHTS)
        trainer.start_iter = 0

    return trainer.train()

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
