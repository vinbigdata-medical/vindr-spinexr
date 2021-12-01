# Copyright 2021 Medical Imaging Center, Vingroup Big Data Insttitute (VinBigdata), Vietnam
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import copy
import numpy as np
import torch
import cv2

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.utils.registry import Registry

from spine.augments import distort_image_with_autoaugment


DATA_MAPPER_REGISTRY = Registry("DATA_MAPPER")

__all__ = ["DATA_MAPPER_REGISTRY", "SpineDatasetMapper", "SpineAutoAugmentMapper"]


@DATA_MAPPER_REGISTRY.register()
class SpineDatasetMapper:
    def __init__(self, cfg, mode):
        """
        "no_transform" mode is used for TTA 
        where images transformation is responsibilty of the model
        """
        assert mode in ["train", "test", "no_transform"]
        self.cfg = cfg
        self.augmentations = self.build_transforms(mode)
        self.mode = mode
        
    def build_transforms(self, mode):
        if mode == "no_transform":
            return [T.NoOpTransform()]
        
        transforms = []
        cfg = self.cfg
        if mode == "train":
            min_size = cfg.INPUT.MIN_SIZE_TRAIN
            max_size = cfg.INPUT.MAX_SIZE_TRAIN
            sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        elif mode == "test":
            min_size = cfg.INPUT.MIN_SIZE_TEST
            max_size = cfg.INPUT.MAX_SIZE_TEST
            sample_style = "choice"
        
        if mode == "train":
            transforms.append(T.RandomFlip())
        
        transforms.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
        if mode == "train":
            print(f"#### training transform {transforms}")
        
        return transforms

    def __call__(self, dataset_dict):
        
        dataset_dict = copy.deepcopy(dataset_dict)

        image = cv2.imread(dataset_dict["file_name"])
        utils.check_image_size(dataset_dict, image)
        image_shape = image.shape[:2] # h,w

        image, transforms = T.apply_transform_gens(self.augmentations, image)

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(
            image.transpose(2,0,1)).astype(np.float32))
        
        annos = [
            utils.transform_instance_annotations(obj, transforms, image_shape)
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]

        instances = utils.annotations_to_instances(
            annos, image_shape
        )
        dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict


@DATA_MAPPER_REGISTRY.register()
class SpineAutoAugmentMapper(SpineDatasetMapper):
    """ augmentation by autoaugment """
    def __call__(self, dataset_dict):
        auto_augment_version =self.cfg.SPINE.AUTO_AUGMENT
        dataset_dict = copy.deepcopy(dataset_dict)

        image = cv2.imread(dataset_dict["file_name"])
        utils.check_image_size(dataset_dict, image)
        
        image, transforms = T.apply_transform_gens(self.augmentations, image)
        image_shape = image.shape[:2] # h,w

        annos = [
            utils.transform_instance_annotations(obj, transforms, image_shape)
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        
        # use auto_augment
        if self.mode == "train":
            # get all box and convert to YXYX_REL
            boxes = [copy.deepcopy(obj['bbox']) for obj in annos]
            boxes = np.array(boxes)[:,[1,0,3,2]]
            h,w,_ = image.shape
            hwhw = np.array([h,w,h,w]).astype(np.float32)
            boxes = boxes/hwhw
            for i in range(10):
                try:
                    new_image, boxes = distort_image_with_autoaugment(image.copy(), boxes.copy(), auto_augment_version)
                    break
                except ValueError as e:
                    if i < 9: 
                        continue
                    else:
                        print(e)
                        print(dataset_dict)
                        raise ValueError(f"failed to apply augment on the image after {i} attempts")
            # convert boxes to XYXY_ABS
            if len(boxes):
                image = new_image
                boxes = (boxes * hwhw)[:,[1,0,3,2]]
                for obj, box in zip(annos, boxes):
                    obj["bbox"] = box
            
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(
            image.transpose(2,0,1)).astype(np.float32))
        
        instances = utils.annotations_to_instances(
            annos, image_shape
        )
        dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict
