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


from typing import Dict, List, Tuple
import torch.nn as nn
import torch
from torch import Tensor

from detectron2.structures import ImageList
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone

from .head import build_classifier_head
from .losses import get_loss_dict

@META_ARCH_REGISTRY.register()
class Classifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        
        self.pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        self.pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.num_classes = len(cfg.MODEL.CLASSIFIER.CLASSES)
        self.head_in_features = cfg.MODEL.CLASSIFIER.IN_FEATURES

        self.backbone = build_backbone(cfg)
        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in cfg.MODEL.CLASSIFIER.IN_FEATURES]
        self.size_divisibility = max([feat.stride for feat in feature_shapes])
        self.head = build_classifier_head(cfg, feature_shapes)
        
        self.loss_names = cfg.MODEL.CLASSIFIER.LOSS_NAMES
        self.loss_weights = cfg.MODEL.CLASSIFIER.LOSS_WEIGHTS
        self.loss_fns = get_loss_dict(cfg)

    def forward(self, batched_inputs: Tuple[Dict[str, Tensor]]):
        """
        batched_inputs: N-length list of data_dict with following items:
            "image": image tensor of shape (C,H,W)
            "classes": binary tensor of shape (M,) where M is number of classes # for training only
        
        return 
            if eval mode:
                prob tensor of shape (N,M)
            if training:
                dict of losses:
                    "<<loss_name>>": loss value
        """
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.head_in_features]
        logits = self.head(features, images.image_sizes)
        
        if self.training:
            targets = torch.stack([x["classes"].to(self.device) for x in batched_inputs])
            return self.loss(logits, targets)
        
        else:
            return self.inference(logits)
    
    def preprocess_image(self, batched_inputs: Tuple[Dict[str, Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        return images

    def loss(self, logits, targets):
        """
        logits: dict["<<feature_map/logit name>>": tensor of shape (N,M)]
        targets: binary tensor of shape (N,M)
        """
        losses = {}
        for (loss_name, loss_fn), weight in zip(self.loss_fns.items(), self.loss_weights):
            loss = 0.0
            for logit in logits:
                loss = loss + loss_fn(logit, targets)
            losses[loss_name] = weight * loss
        
        return losses
        
    def inference(self, logits):
        """
        logits: dict["<<feature_map/logit name>>": tensor of shape (N,M)]
        
        return prob of shape (N,M)
        """
        
        stacked_logits = torch.stack(logits)
        scores = stacked_logits.mean(dim=0).sigmoid_()
        return scores


if __name__ == "__main__":
    from detectron2.config import get_cfg
    from config import add_classifier_config
    cfg = get_cfg()
    add_classifier_config(cfg)

    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.PIXEL_MEAN = [0.,0.,0.]
    cfg.MODEL.PIXEL_STD = [1.,1.,1.]

    cfg.MODEL.CLASSIFIER.IN_FEATURES = ["res4", "res5"]
    cfg.MODEL.CLASSIFIER.CLASSES = ["test1"]
    cfg.MODEL.CLASSIFIER.HEAD_NAME = "SimpleHead"
    cfg.MODEL.CLASSIFIER.LOSS_NAMES = ["BCE", "focal"]
    cfg.MODEL.CLASSIFIER.LOSS_WEIGHTS = [1.0, 0.5]
    cfg.MODEL.CLASSIFIER.FOCAL_ALPHA = -1
    cfg.MODEL.CLASSIFIER.FOCAL_GAMMA = 2
    cfg.MODEL.CLASSIFIER.LOSS_REDUCTION = "mean"

    cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"
    cfg.MODEL.RESNETS.OUT_FEATURES = ["res4", "res5"]
    # input_shapes = [ShapeSpec(channels=8, stride=8), ShapeSpec(channels=16, stride=16)]
    # sizes = [(256,512), (378,768)]
    # inputs = [torch.randn((2,8,128,128)), torch.randn((2,16,64,64))]
    # heads = build_classifier_head(cfg, input_shapes)
    # print(heads(inputs, sizes))
    model = Classifier(cfg)
    model.train()
    # model.eval()

    print(model([{"image": torch.randn((3,512,512)), "classes": torch.tensor([1.])}]))