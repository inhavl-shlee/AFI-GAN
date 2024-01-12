# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import pickle
from fvcore.common.checkpoint import Checkpointer
from fvcore.common.file_io import PathManager

import detectron2.utils.comm as comm
import logging
import copy
import re
import torch

from detectron2.checkpoint.c2_model_loading import align_and_update_state_dicts

class AF_DetectionCheckpointer(Checkpointer):
    """
    Same as :class:`Checkpointer`, but is able to handle models in detectron & detectron2
    model zoo, and apply conversions for legacy models.
    """

    def __init__(self, model, save_dir="", *, save_to_disk=None, **checkpointables):
        is_main_process = comm.is_main_process()
        super().__init__(
            model,
            save_dir,
            save_to_disk=is_main_process if save_to_disk is None else save_to_disk,
            **checkpointables,
        )

    def _load_file(self, filename):
        if filename.endswith(".pkl"):
            with PathManager.open(filename, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            if "model" in data and "__author__" in data:
                # file is in Detectron2 model zoo format
                self.logger.info("Reading a file from '{}'".format(data["__author__"]))
                return data
            else:
                # assume file is from Caffe2 / Detectron1 model zoo
                if "blobs" in data:
                    # Detection models have "blobs", but ImageNet models don't
                    data = data["blobs"]
                data = {k: v for k, v in data.items() if not k.endswith("_momentum")}
                return {"model": data, "__author__": "Caffe2", "matching_heuristics": True}

        loaded = super()._load_file(filename)  # load native pth checkpoint
        if "model" not in loaded:
            loaded = {"model": loaded}
        return loaded

    def _load_model(self, checkpoint):
        if checkpoint.get("matching_heuristics", False):
            self._convert_ndarray_to_tensor(checkpoint["model"])
            # convert weights by name-matching heuristics
            model_state_dict = self.model.state_dict()
            align_and_update_state_dicts(
                model_state_dict,
                checkpoint["model"],
                c2_conversion=checkpoint.get("__author__", None) == "Caffe2",
            )
            checkpoint["model"] = model_state_dict
        # for non-caffe2 models, use standard ways to load it
        super()._load_model(checkpoint)

    def _load_AFExtractor_weights_file(self, filename):
        checkpoint = self._load_file(filename)
        model_state_dict = self.model.state_dict()
        self.align_and_update_state_dicts_AFExtractor(model_state_dict, checkpoint["model"])
        checkpoint["model"] = model_state_dict
        self._load_model(checkpoint)

    def _load_TargetDetector_weights_file(self, filename):
        checkpoint = self._load_file(filename)
        model_state_dict = self.model.state_dict()
        self.align_and_update_state_dicts_TargetDetector(model_state_dict, checkpoint["model"])
        checkpoint["model"] = model_state_dict
        self._load_model(checkpoint)

    def convert_AFI_names(self, weights):
        """
        Map Caffe2 Detectron weight names to Detectron2 names.

        Args:
            weights (dict): name -> tensor

        Returns:
            dict: detectron2 names -> tensor
            dict: detectron2 names -> C2 names
        """
        logger = logging.getLogger(__name__)
        logger.info("Remapping Generator weights ......")
        original_keys = sorted(weights.keys())
        layer_keys = copy.deepcopy(original_keys)

        layer_keys = [k.replace("Generators", "backbone.afi_module.Generators") for k in layer_keys]

        # --------------------------------------------------------------------------
        # Done with replacements
        # --------------------------------------------------------------------------
        assert len(set(layer_keys)) == len(layer_keys)
        assert len(original_keys) == len(layer_keys)

        new_weights = {}
        new_keys_to_original_keys = {}
        for orig, renamed in zip(original_keys, layer_keys):
            new_keys_to_original_keys[renamed] = orig

            new_weights[renamed] = weights[orig]

        return new_weights, new_keys_to_original_keys

    def remain_only_AFI_names(self, weights):
        logger = logging.getLogger(__name__)
        logger.info("Remapping AF Extractor weights ......")
        original_keys = sorted(weights.keys())
        layer_keys = copy.deepcopy(original_keys)

        new_weights = {}
        new_keys_to_original_keys = {}
        for orig, renamed in zip(original_keys, layer_keys):
            if re.search('afi_module', renamed):
                new_keys_to_original_keys[renamed] = orig

                new_weights[renamed] = weights[orig]

        return new_weights, new_keys_to_original_keys

    def align_and_update_state_dicts_AFExtractor(self, model_state_dict, ckpt_state_dict):

        model_keys = sorted(model_state_dict.keys())

        # original_keys = {x: x for x in ckpt_state_dict.keys()}
        ckpt_state_dict, original_keys = self.convert_AFI_names(ckpt_state_dict)

        ckpt_keys = sorted(ckpt_state_dict.keys())

        def match(a, b):
            # Matched ckpt_key should be a complete (starts with '.') suffix.
            # For example, roi_heads.mesh_head.whatever_conv1 does not match conv1,
            # but matches whatever_conv1 or mesh_head.whatever_conv1.
            return a == b or a.endswith("." + b)

        # get a matrix of string matches, where each (i, j) entry correspond to the size of the
        # ckpt_key string, if it matches
        match_matrix = [len(j) if match(i, j) else 0 for i in model_keys for j in ckpt_keys]
        match_matrix = torch.as_tensor(match_matrix).view(len(model_keys), len(ckpt_keys))
        # use the matched one with longest size in case of multiple matches
        max_match_size, idxs = match_matrix.max(1)
        # remove indices that correspond to no-match
        idxs[max_match_size == 0] = -1

        # used for logging
        max_len_model = max(len(key) for key in model_keys) if model_keys else 1
        max_len_ckpt = max(len(key) for key in ckpt_keys) if ckpt_keys else 1
        log_str_template = "{: <{}} loaded from {: <{}} of shape {}"
        logger = logging.getLogger(__name__)
        # matched_pairs (matched checkpoint key --> matched model key)
        matched_keys = {}
        for idx_model, idx_ckpt in enumerate(idxs.tolist()):
            if idx_ckpt == -1:
                continue
            key_model = model_keys[idx_model]
            key_ckpt = ckpt_keys[idx_ckpt]
            value_ckpt = ckpt_state_dict[key_ckpt]
            shape_in_model = model_state_dict[key_model].shape

            if shape_in_model != value_ckpt.shape:
                logger.warning(
                    "Shape of {} in checkpoint is {}, while shape of {} in model is {}.".format(
                        key_ckpt, value_ckpt.shape, key_model, shape_in_model
                    )
                )
                logger.warning(
                    "{} will not be loaded. Please double check and see if this is desired.".format(
                        key_ckpt
                    )
                )
                continue

            model_state_dict[key_model] = value_ckpt.clone()
            if key_ckpt in matched_keys:  # already added to matched_keys
                logger.error(
                    "Ambiguity found for {} in checkpoint!"
                    "It matches at least two keys in the model ({} and {}).".format(
                        key_ckpt, key_model, matched_keys[key_ckpt]
                    )
                )
                raise ValueError("Cannot match one checkpoint key to multiple keys in the model.")

            matched_keys[key_ckpt] = key_model
            logger.info(
                log_str_template.format(
                    key_model,
                    max_len_model,
                    original_keys[key_ckpt],
                    max_len_ckpt,
                    tuple(shape_in_model),
                )
            )

    def align_and_update_state_dicts_TargetDetector(self, model_state_dict, ckpt_state_dict):

        model_keys = sorted(model_state_dict.keys())

        # original_keys = {x: x for x in ckpt_state_dict.keys()}
        ckpt_state_dict, original_keys = self.remain_only_AFI_names(ckpt_state_dict)

        ckpt_keys = sorted(ckpt_state_dict.keys())

        def match(a, b):
            # Matched ckpt_key should be a complete (starts with '.') suffix.
            # For example, roi_heads.mesh_head.whatever_conv1 does not match conv1,
            # but matches whatever_conv1 or mesh_head.whatever_conv1.
            return a == b or a.endswith("." + b)

        # get a matrix of string matches, where each (i, j) entry correspond to the size of the
        # ckpt_key string, if it matches
        match_matrix = [len(j) if match(i, j) else 0 for i in model_keys for j in ckpt_keys]
        match_matrix = torch.as_tensor(match_matrix).view(len(model_keys), len(ckpt_keys))
        # use the matched one with longest size in case of multiple matches
        max_match_size, idxs = match_matrix.max(1)
        # remove indices that correspond to no-match
        idxs[max_match_size == 0] = -1

        # used for logging
        max_len_model = max(len(key) for key in model_keys) if model_keys else 1
        max_len_ckpt = max(len(key) for key in ckpt_keys) if ckpt_keys else 1
        log_str_template = "{: <{}} loaded from {: <{}} of shape {}"
        logger = logging.getLogger(__name__)
        # matched_pairs (matched checkpoint key --> matched model key)
        matched_keys = {}
        for idx_model, idx_ckpt in enumerate(idxs.tolist()):
            if idx_ckpt == -1:
                continue
            key_model = model_keys[idx_model]
            key_ckpt = ckpt_keys[idx_ckpt]
            value_ckpt = ckpt_state_dict[key_ckpt]
            shape_in_model = model_state_dict[key_model].shape

            if shape_in_model != value_ckpt.shape:
                logger.warning(
                    "Shape of {} in checkpoint is {}, while shape of {} in model is {}.".format(
                        key_ckpt, value_ckpt.shape, key_model, shape_in_model
                    )
                )
                logger.warning(
                    "{} will not be loaded. Please double check and see if this is desired.".format(
                        key_ckpt
                    )
                )
                continue

            model_state_dict[key_model] = value_ckpt.clone()
            if key_ckpt in matched_keys:  # already added to matched_keys
                logger.error(
                    "Ambiguity found for {} in checkpoint!"
                    "It matches at least two keys in the model ({} and {}).".format(
                        key_ckpt, key_model, matched_keys[key_ckpt]
                    )
                )
                raise ValueError("Cannot match one checkpoint key to multiple keys in the model.")

            matched_keys[key_ckpt] = key_model
            logger.info(
                log_str_template.format(
                    key_model,
                    max_len_model,
                    original_keys[key_ckpt],
                    max_len_ckpt,
                    tuple(shape_in_model),
                )
            )