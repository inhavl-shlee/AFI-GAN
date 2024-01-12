# Baseline code : defaults.py from detectron2 #

import argparse
import logging
import os
from collections import OrderedDict
import torch
import time
import numpy as np
from fvcore.common.file_io import PathManager
from fvcore.nn.precise_bn import get_bn_modules
from torch.nn.parallel import DistributedDataParallel

import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils import comm
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
from detectron2.utils.logger import setup_logger

from detectron2.engine import hooks
from detectron2.engine.train_loop import SimpleTrainer, TrainerBase

from afigan.modeling import build_guide_model
from afigan.modeling.feat_interpol import generator_rdb as G_rdb
from afigan.modeling.feat_interpol import feature_patch_discriminator as D
from afigan.engine.checkpoint import AF_DetectionCheckpointer

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from detectron2.structures import ImageList
import math
__all__ = ["Multi_Scale_AF_Extractor_Trainer"]

import pickle



class Multi_Scale_AF_Extractor_Trainer(SimpleTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        logger = logging.getLogger("af-extractor")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        self.device = cfg.MODEL.DEVICE
        # Assume these objects must be constructed in this order.
        model, D_model, feature_model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        feature_model_optimizer = self.build_optimizer(cfg, feature_model)
        feature_model_checkpointer = DetectionCheckpointer(feature_model,
                                                           optimizer=feature_model_optimizer)

        with open(cfg.MODEL.GUIDE_WEIGHTS, 'rb') as f:
            obj = f.read()
        checkpoint = pickle.loads(obj, encoding='latin1')

        feature_model_checkpointer._load_model(checkpoint)

        data_loader = self.build_train_loader(cfg)

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )
            D_model = DistributedDataParallel(
                D_model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )
            feature_model = DistributedDataParallel(
                feature_model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        self.distributed = comm.get_world_size() > 1

        super().__init__(model, data_loader, optimizer)

        if self.distributed:
            D_optimizer = self.build_optimizer(cfg, D_model.module.Discriminators[0])
        else:
            D_optimizer = self.build_optimizer(cfg, D_model.Discriminators[0])

        self.D_optimizer = D_optimizer
        self.D_scheduler = self.build_lr_scheduler(cfg, D_optimizer)

        self.D_model = D_model
        self.feature_model = feature_model
        os.makedirs(os.path.join(cfg.OUTPUT_DIR, 'Disc'), exist_ok=True)
        D_output_dir = os.path.join(cfg.OUTPUT_DIR, 'Disc')
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        # Assume no other objects need to be checkpointed.
        # We can later make it checkpoint the stateful hooks
        self.D_checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            D_model,
            D_output_dir,
            optimizer=D_optimizer,
            scheduler=self.D_scheduler,
        )
        checkpoint = torch.load(
            cfg.MODEL.AFI_DIS_WEIGHTS,
            map_location=torch.device('cpu'))

        del checkpoint["optimizer"]
        del checkpoint["scheduler"]
        del checkpoint["iteration"]
        self.D_checkpointer._load_model(checkpoint)

        self.checkpointer = AF_DetectionCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )

        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.discriminator_criterion = nn.BCEWithLogitsLoss()

        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True`, and last checkpoint exists, resume from it.

        Otherwise, load a model specified by the config.

        Args:
            resume (bool): whether to do resume or not
        """
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration (or iter zero if there's no checkpoint).
        self.start_iter = (
                self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume).get(
                    "iteration", -1
                )
                + 1
        )
    def load_AFIGEN_weight(self, weight_path):
        if weight_path.endswith(".pth"):
            self.checkpointer._load_AFExtractor_weights_file(weight_path)
        else:
            raise NotImplementedError("Not Implemented Error")


    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.D_model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.D_model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))
            ret.append(hooks.PeriodicCheckpointer(self.D_checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    def build_writers(self):
        """
        Build a list of writers to be used. By default it contains
        writers that write metrics to the screen,
        a json file, and a tensorboard event file respectively.
        If you'd like a different list of writers, you can overwrite it in
        your trainer.

        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.

        It is now implemented by:

        .. code-block:: python

            return [
                CommonMetricPrinter(self.max_iter),
                JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
                TensorboardXWriter(self.cfg.OUTPUT_DIR),
            ]

        """
        # Here the default print/log frequency of each writer is used.
        return [
            # It may not always print what you want to see, since it prints "common" metrics only.
            CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(self.cfg.OUTPUT_DIR),
        ]

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """

        self.feature_model.eval()

        if self.distributed:
            self.D_model.module.Discriminators[0].train()

        else:
            self.D_model.Discriminators[0].train()

        self.model.train()

        super().train(self.start_iter, self.max_iter)
        if len(self.cfg.TEST.EXPECTED_RESULTS) and comm.is_main_process():
            assert hasattr(
                self, "_last_eval_results"
            ), "No evaluation results obtained during training!"
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results



    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        hr_feature_ = self.feature_model(data, img_dict_name='image')

        """
        If you want to do something with the losses, you can wrap the model.
        """
        loss_dict, upsampled_feature_ = self.model(data)

        hr_features = []
        uphr_features = []
        for idx in range(len(hr_feature_)):
            for d in range(2, 7):
                hr_features.append(torch.nn.functional.interpolate(hr_feature_[0]['features'][f'p{d}'].detach(), scale_factor=0.5))
                uphr_features.append(upsampled_feature_[0]['features'][f'p{d}'])

        d_loss_dict = dict()
        for p_lv, (hr_feature, uphr_feature) in enumerate(zip(hr_features, uphr_features), 2):
            hr_feature = self._reshape_feature(hr_feature, uphr_feature.size())
            uphr_feature = self._reshape_feature(uphr_feature, hr_feature.size())

            if self.distributed:
                logit_real = self.D_model.module.Discriminators[0](hr_feature)
                logit_fake = self.D_model.module.Discriminators[0](uphr_feature.detach())
            else:
                logit_real = self.D_model.Discriminators[0](hr_feature)
                logit_fake = self.D_model.Discriminators[0](uphr_feature.detach())

            real = Variable(torch.ones(logit_real.size())).to(self.device)
            fake = Variable(torch.zeros(logit_fake.size())).to(self.device)

            tmp_d_loss = {f'd_loss_p{p_lv}': self.discriminator_criterion(logit_real, real) +
                                             self.discriminator_criterion(logit_fake, fake)}
            # print(tmp_d_loss)
            d_loss_dict.update(tmp_d_loss)

        d_losses = sum(d_loss_dict.values())
        self._detect_anomaly(d_losses, d_loss_dict)

        metrics_dict = d_loss_dict
        metrics_dict["D_data_time"] = data_time
        self._write_metrics(metrics_dict)
        """
        If you need to accumulate gradients or something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.D_optimizer.zero_grad()
        d_losses.backward()

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method.
        """
        self.D_optimizer.step()

        g_loss_dict = dict()
        for p_lv, (hr_feature, uphr_feature) in enumerate(zip(hr_features, uphr_features), 2):
            hr_feature = self._reshape_feature(hr_feature, uphr_feature.size())
            uphr_feature = self._reshape_feature(uphr_feature, hr_feature.size())

            if self.distributed:
                logit_fake = self.D_model.module.Discriminators[0](uphr_feature).detach()
                logit_real = self.D_model.module.Discriminators[0](hr_feature)
            else:
                logit_fake = self.D_model.Discriminators[0](uphr_feature).detach()
                logit_real = self.D_model.Discriminators[0](hr_feature)

            real = Variable(torch.ones(logit_real.size())).to(self.device)
            tmp_adv_loss = {f'adv_loss_p{p_lv}': self.discriminator_criterion(logit_fake, real)}

            tmp_content_loss = {f'content_loss_p{p_lv}': F.l1_loss(uphr_feature, hr_feature)}

            g_loss_dict.update({f'g_loss_p{p_lv}': (tmp_adv_loss[f'adv_loss_p{p_lv}'] * (1e-3) + tmp_content_loss[
                f'content_loss_p{p_lv}'])})

        loss_dict.update(g_loss_dict)

        losses = sum(loss_dict.values())
        self._detect_anomaly(losses, loss_dict)

        metrics_dict = loss_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        """
        If you need to accumulate gradients or something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        losses.backward()

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method.
        """
        self.optimizer.step()




    def _detect_anomaly(self, losses, loss_dict):
        if not torch.isfinite(losses).all():
            raise FloatingPointError(
                "Loss became infinite or NaN at iteration={}!\nloss_dict = {}".format(
                    self.iter, loss_dict
                )
            )

    def _write_metrics(self, metrics_dict: dict):
        """
        Args:
            metrics_dict (dict): dict of scalar metrics
        """
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)
            if "G_data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("G_data_time") for x in all_metrics_dict])
                self.storage.put_scalar("G_data_time", data_time)
            if "D_data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("D_data_time") for x in all_metrics_dict])
                self.storage.put_scalar("D_data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(loss for loss in metrics_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """

        model = build_model(cfg)

        D_model = D.Discriminator().to(torch.device(cfg.MODEL.DEVICE))
        feature_model = build_guide_model(cfg).to(torch.device(cfg.MODEL.DEVICE))

        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        logger.info("D_Model:\n{}".format(D_model))
        logger.info("feature_Model:\n{}".format(feature_model))

        return model, D_model, feature_model

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Returns:
            torch.optim.Optimizer:

        It now calls :func:`detectron2.solver.build_optimizer`.
        Overwrite it if you'd like a different optimizer.
        """
        return build_optimizer(cfg, model)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """

        return build_af_extractor_train_loader(cfg)

    def _reshape_feature(self, tr_feature, hr_feature_size):
        if hr_feature_size[2] != tr_feature.size()[2] or hr_feature_size[3] != tr_feature.size()[3]:
            _W = hr_feature_size[2] if hr_feature_size[2] < tr_feature.size()[2] else tr_feature.size()[2]
            _H = hr_feature_size[3] if hr_feature_size[3] < tr_feature.size()[3] else tr_feature.size()[3]
            return tr_feature[:, :, 0:_W, 0:_H]
        return tr_feature


import operator
from detectron2.utils.comm import get_world_size
from detectron2.data.build import get_detection_dataset_dicts
from detectron2.data.common import AspectRatioGroupedDataset, DatasetFromList, MapDataset
from detectron2.data.build import trivial_batch_collator, worker_init_reset_seed
from detectron2.data import samplers

from afigan.engine.dataset_mapper import DatasetMapper as DatasetMapper_afextractor

def build_af_extractor_train_loader(cfg, mapper=None):
    """
    A data loader is created by the following steps:

    1. Use the dataset names in config to query :class:`DatasetCatalog`, and obtain a list of dicts.
    2. Start workers to work on the dicts. Each worker will:

       * Map each metadata dict into another format to be consumed by the model.
       * Batch them by simply putting dicts into a list.

    The batched ``list[mapped_dict]`` is what this dataloader will return.

    Args:
        cfg (CfgNode): the config
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            By default it will be `DatasetMapper(cfg, True)`.

    Returns:
        an infinite iterator of training data
    """
    num_workers = get_world_size()
    images_per_batch = cfg.SOLVER.IMS_PER_BATCH
    assert (
            images_per_batch % num_workers == 0
    ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of workers ({}).".format(
        images_per_batch, num_workers
    )
    assert (
            images_per_batch >= num_workers
    ), "SOLVER.IMS_PER_BATCH ({}) must be larger than the number of workers ({}).".format(
        images_per_batch, num_workers
    )
    images_per_worker = images_per_batch // num_workers

    dataset_dicts = get_detection_dataset_dicts(
        cfg.DATASETS.TRAIN,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
    )
    dataset = DatasetFromList(dataset_dicts, copy=False)

    if mapper is None:
        mapper = DatasetMapper_afextractor(cfg, [0.5], True)
    dataset = MapDataset(dataset, mapper)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))
    if sampler_name == "TrainingSampler":
        sampler = samplers.TrainingSampler(len(dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        sampler = samplers.RepeatFactorTrainingSampler(
            dataset_dicts, cfg.DATALOADER.REPEAT_THRESHOLD
        )
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))

    if cfg.DATALOADER.ASPECT_RATIO_GROUPING:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            batch_sampler=None,
            collate_fn=operator.itemgetter(0),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        data_loader = AspectRatioGroupedDataset(data_loader, images_per_worker)
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_worker, drop_last=True
        )
        # drop_last so the batch always have the same size
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            batch_sampler=batch_sampler,
            collate_fn=trivial_batch_collator,
            worker_init_fn=worker_init_reset_seed,
        )

    return data_loader