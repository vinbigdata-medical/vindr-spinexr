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


from functools import partial 
from collections import OrderedDict

import torch.nn as nn
from fvcore.nn import sigmoid_focal_loss
import torch.nn.functional as F


def get_loss_dict(cfg):
    loss_fn = {
        "focal": partial(sigmoid_focal_loss, 
            alpha=cfg.MODEL.CLASSIFIER.FOCAL_ALPHA,
            gamma=cfg.MODEL.CLASSIFIER.FOCAL_GAMMA,
            reduction=cfg.MODEL.CLASSIFIER.LOSS_REDUCTION,
        ),

        "BCE": nn.BCEWithLogitsLoss(reduction=cfg.MODEL.CLASSIFIER.LOSS_REDUCTION),
    }
    loss_fns = OrderedDict()
    for name in cfg.MODEL.CLASSIFIER.LOSS_NAMES:
        loss_fns[name] = loss_fn[name]
    return loss_fns