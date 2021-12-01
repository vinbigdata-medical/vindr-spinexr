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


from typing import List
import math
import torch.nn as nn
import torch

from detectron2.utils.registry import Registry
from detectron2.layers import ShapeSpec


CLASSIFIER_HEAD_REGISTRY = Registry("CLASSIFIER_HEAD")


def build_classifier_head(cfg, input_shape: List[ShapeSpec]):
    head_name = cfg.MODEL.CLASSIFIER.HEAD_NAME
    head = CLASSIFIER_HEAD_REGISTRY.get(head_name)(cfg, input_shape)

    return head


@CLASSIFIER_HEAD_REGISTRY.register()
class SimpleHead(nn.Module):
    """
    
    """
    def __init__(self, cfg, input_shapes):
        super().__init__()
        self.input_shapes = input_shapes
        n_classes = len(cfg.MODEL.CLASSIFIER.CLASSES)
        heads = []
        for input_shape in input_shapes:
            heads.append(nn.Linear(input_shape.channels, n_classes))
        
        self.heads = nn.ModuleList(heads)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()

    def forward(self, features, image_sizes):
        """
        features: list of feature maps
        image_sizes: list of (H,W)

        return list of logits of shape (N,M) where M is number of classes
        """
        logits = []
        for feature, head, shape in zip(features, self.heads, self.input_shapes):
            pooled_feature = self.flatten(self.avg_pool(feature))
            logits.append(head(pooled_feature))

        return logits


if __name__ == "__main__":
    class CFG:
        pass

    cfg = CFG()
    cfg.MODEL = CFG()
    cfg.MODEL.CLASSIFIER = CFG()
    cfg.MODEL.CLASSIFIER.CLASSES = ["test1"]
    cfg.MODEL.CLASSIFIER.HEAD_NAME = "SimpleHead"

    input_shapes = [ShapeSpec(channels=8, stride=8), ShapeSpec(channels=16, stride=16)]
    sizes = [(256,512), (378,768)]
    inputs = [torch.randn((2,8,128,128)), torch.randn((2,16,64,64))]
    heads = build_classifier_head(cfg, input_shapes)
    print(heads(inputs, sizes))