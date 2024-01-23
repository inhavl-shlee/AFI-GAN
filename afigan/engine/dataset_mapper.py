##
import copy
import logging
import numpy as np
import torch
from fvcore.common.file_io import PathManager
from PIL import Image

from detectron2.data import detection_utils as utils
from afigan.engine import afigan_utils as utils_gen
from detectron2.data import transforms as T
from afigan.engine import transform_gen as T_gen



"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapper"]


class DatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __init__(self, cfg, scale_ratio=[0.5],is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            logging.getLogger(__name__).info("CropGen used in training: " + str(self.crop_gen))
        else:
            self.crop_gen = None

        self.tfm_gens = utils_gen.build_transform_gen(cfg, is_train)
        self.tfm_gens_ratio = copy.deepcopy(self.tfm_gens)

        # fmt: off
        self.img_format     = cfg.INPUT.FORMAT
        self.mask_on        = cfg.MODEL.MASK_ON
        self.mask_format    = cfg.INPUT.MASK_FORMAT
        self.keypoint_on    = cfg.MODEL.KEYPOINT_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        self.scale_ratio = scale_ratio
        # fmt: on

        self.keypoint_hflip_indices = None

        if self.load_proposals:
            self.min_box_side_len = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        self.is_train = is_train

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils_gen.read_image(dataset_dict["file_name"], format=self.img_format)
        image_r = copy.deepcopy(image)

        utils_gen.check_image_size(dataset_dict, image)


        if "annotations" not in dataset_dict:
            image, transforms = T_gen.apply_transform_gens(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image
            )

            image_r, transforms_r = T_gen.apply_transform_gens_overlap2(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens_ratio, image_r, transforms
            )
        else:
            # Crop around an instance if there are instances in the image.
            # USER: Remove if you don't use cropping
            if self.crop_gen:
                crop_tfm = utils_gen.gen_crop_transform_with_instance(
                    self.crop_gen.get_crop_size(image.shape[:2]),
                    image.shape[:2],
                    np.random.choice(dataset_dict["annotations"]),
                )
                image = crop_tfm.apply_image(image)
            image, transforms = T_gen.apply_transform_gens(self.tfm_gens, image)

            image_r, transforms_r = T_gen.apply_transform_gens_overlap2(self.tfm_gens_ratio, image_r, transforms)

            if self.crop_gen:
                transforms = crop_tfm + transforms

        image_shape = image.shape[:2]  # h, w
        image_r_shape = image_r.shape[:2]
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.

        image = image.transpose(2, 0, 1)
        image_r = image_r.transpose(2,0,1)

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image))
        dataset_dict["image_x0.5"] = torch.as_tensor(np.ascontiguousarray(image_r))
        for ratio in self.scale_ratio:
            dataset_dict["width_x{}".format(ratio)],dataset_dict["heigth_x{}".format(ratio)] = int(image.shape[1]*ratio), int(image.shape[2]*ratio)
        # USER: Remove if you don't use pre-computed proposals.
        if self.load_proposals:
            utils_gen.transform_proposals(
                dataset_dict, image_shape, transforms, self.min_box_side_len, self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)

            for ratio in self.scale_ratio:
                dataset_dict.pop("image_x{}".format(ratio),None)

            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)

            annotations_ = dataset_dict.pop("annotations")
            annotations_copy = copy.deepcopy(annotations_)


            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils_gen.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in annotations_
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils_gen.annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format
            )
            # Create a tight bounding box from masks, useful when image is cropped
            if self.crop_gen and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils_gen.filter_empty_instances(instances)

            for ratio in self.scale_ratio:
                annos2 = [
                    utils_gen.transform_instance_annotations(
                        obj, transforms_r, image_r_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                    )
                    for obj in annotations_copy
                    if obj.get("iscrowd", 0) == 0
                ]
                instances2 = utils_gen.annotations_to_instances(
                    annos2, image_r_shape, mask_format=self.mask_format
                )
                # Create a tight bounding box from masks, useful when image is cropped
                if self.crop_gen and instances2.has("gt_masks"):
                    instances2.gt_boxes = instances2.gt_masks.get_bounding_boxes()
                dataset_dict[f"instances_x{ratio}"] = utils_gen.filter_empty_instances(instances2)


        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            with PathManager.open(dataset_dict.pop("sem_seg_file_name"), "rb") as f:
                sem_seg_gt = Image.open(f)
                sem_seg_gt = np.asarray(sem_seg_gt, dtype="uint8")
            sem_seg_gt = transforms.apply_segmentation(sem_seg_gt)
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            dataset_dict["sem_seg"] = sem_seg_gt
        return dataset_dict
