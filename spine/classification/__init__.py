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


from .classifiers import Classifier
from .cls_dataset_dict import setup_cls_data_catalog, SpineClsDatasetFunction
from .dataloader import build_classification_test_loader, build_classification_train_loader
from .cls_dataset_mapper import SpineClsDatasetMapper, SpineClsDatasetMapper2
from .head import SimpleHead, CLASSIFIER_HEAD_REGISTRY, build_classifier_head
from .config import add_classifier_config
from .classification_evaluator import ClassificationEvaluator

from .backbone import DenseNet