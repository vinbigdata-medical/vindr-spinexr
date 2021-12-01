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

from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils

from spine.dataset_mapper import DATA_MAPPER_REGISTRY


@DATA_MAPPER_REGISTRY.register()
class SpineClsDatasetMapper:
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

        image, transforms = T.apply_transform_gens(self.augmentations, image)

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(
            image.transpose(2,0,1)).astype(np.float32))
        dataset_dict["classes"] = torch.as_tensor(dataset_dict["classes"])
        
        return dataset_dict


@DATA_MAPPER_REGISTRY.register()
class SpineClsDatasetMapper2(SpineClsDatasetMapper):
    def build_transforms(self, mode):
        if mode == "no_transform":
            return [T.NoOpTransform()]
        
        transforms = []
        cfg = self.cfg
        size = cfg.MODEL.CLASSIFIER.INPUT_SIZE
        if mode == "train":
            transforms.append(T.RandomFlip())
        
        transforms.append(T.Resize((size, size), ))
        if mode == "train":
            print(f"#### training transform {transforms}")
        
        return transforms