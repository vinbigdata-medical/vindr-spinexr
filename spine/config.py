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


from detectron2.config import CfgNode as CN

def add_spine_config(cfg):
    cfg.SPINE = CN()
    cfg.SPINE.FINDINGS = []
    cfg.SPINE.TRAIN_METADATA = ""
    cfg.SPINE.TRAIN_ANNOTATION = ""
    cfg.SPINE.TRAIN_DATA_FOLDER = ""
    cfg.SPINE.TEST_METADATA = ""
    cfg.SPINE.TEST_ANNOTATION = ""
    cfg.SPINE.TEST_DATA_FOLDER = ""
    cfg.SPINE.ARCH = ""
    cfg.SPINE.IOU_THRESHOLD = (0.2,0.5)
    cfg.SPINE.AUTO_AUGMENT = "v1"
    cfg.SPINE.DATA_MAPPER = "SpineDatasetMapper"
    cfg.SPINE.DATASET_DICT_FN = "SpineDatasetFunction"
    cfg.SPINE.INCLUDE_NF = False
    
    cfg.SPINE.TASK = "detection"
