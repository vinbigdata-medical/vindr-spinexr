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


from torchvision import models
import torch.nn as nn

from detectron2.modeling import Backbone, BACKBONE_REGISTRY
from detectron2.layers import ShapeSpec


@BACKBONE_REGISTRY.register()
class DenseNet(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()
        assert input_shape.channels == 3
        depth = cfg.MODEL.DENSENET.DEPTH
        assert depth in [121,161,169,201]
        out_features = cfg.MODEL.DENSENET.OUT_FEATURES
        for name in out_features:
            assert name in ["dense1", "dense2", "dense3", "dense4"]
        self._out_features = out_features
        self._out_feature_strides = {"dense1": 8, "dense2": 16, "dense3": 32, "dense4": 32}
        self._out_feature_channels = {}
        pretrained = cfg.MODEL.CLASSIFIER.PRETRAINED

        if depth == 121:
            _densenet = models.densenet121(pretrained=pretrained)
        elif depth == 161:
            _densenet = models.densenet161(pretrained=pretrained)
        elif depth == 169:
            _densenet = models.densenet169(pretrained=pretrained)
        elif depth == 201:
            _densenet = models.densenet201(pretrained=pretrained)
        self.stem = nn.Sequential(
            _densenet.features.conv0,
            _densenet.features.norm0,
            _densenet.features.relu0,
            _densenet.features.pool0,
        )

        self.block1 = _densenet.features.denseblock1
        self.transition1 = _densenet.features.transition1
        self._out_feature_channels["dense1"] = self.transition1.conv.out_channels

        self.block2 = _densenet.features.denseblock2
        self.transition2 = _densenet.features.transition2
        self._out_feature_channels["dense2"] = self.transition2.conv.out_channels

        self.block3 = _densenet.features.denseblock3
        self.transition3 = _densenet.features.transition3
        self._out_feature_channels["dense3"] = self.transition3.conv.out_channels

        self.block4 = _densenet.features.denseblock4
        self.norm5 = _densenet.features.norm5
        self._out_feature_channels["dense4"] = _densenet.classifier.in_features

        del _densenet

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        x = self.block1(x)
        x = self.transition1(x)
        if "dense1" in self._out_features:
            outputs["dense1"] = x
        
        x = self.block2(x)
        x = self.transition2(x)
        if "dense2" in self._out_features:
            outputs["dense2"] = x
        
        x = self.block3(x)
        x = self.transition3(x)
        if "dense3" in self._out_features:
            outputs["dense3"] = x
        
        x = self.block4(x)
        x = self.norm5(x)
        if "dense4" in self._out_features:
            outputs["dense4"] = x
        
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }