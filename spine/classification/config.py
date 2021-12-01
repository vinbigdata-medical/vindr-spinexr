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


"""
additional configs for classifier
"""

from detectron2.config import CfgNode as CN


def add_classifier_config(cfg):
    cfg.MODEL.CLASSIFIER = CN()
    
    cfg.MODEL.CLASSIFIER.CLASSES = ["Abnormal"]
    cfg.MODEL.CLASSIFIER.NORM = "BN"
    cfg.MODEL.CLASSIFIER.HEAD_NAME = "SimpleHead"
    cfg.MODEL.CLASSIFIER.IN_FEATURES = ["res4", "res5"]
    cfg.MODEL.CLASSIFIER.LOSS_NAMES = ["BCE",]
    cfg.MODEL.CLASSIFIER.LOSS_WEIGHTS = [1.0,]
    cfg.MODEL.CLASSIFIER.FOCAL_ALPHA = -1
    cfg.MODEL.CLASSIFIER.FOCAL_GAMMA = 2
    cfg.MODEL.CLASSIFIER.LOSS_REDUCTION = "mean"
    cfg.MODEL.CLASSIFIER.INPUT_SIZE = 224
    cfg.MODEL.CLASSIFIER.PRETRAINED = True

    cfg.MODEL.DENSENET = CN()
    cfg.MODEL.DENSENET.OUT_FEATURES = [""]
    cfg.MODEL.DENSENET.DEPTH = 121

    cfg.CLASSIFICATION = CN()
    cfg.CLASSIFICATION.BOOTSTRAP = False
    cfg.CLASSIFICATION.BOOTSTRAP_SAMPLES = 10000
    cfg.CLASSIFICATION.BOOTSTRAP_CI = 0.95
    
