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

from collections import defaultdict
import argparse
import torch
import numpy as np

from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.utils.logger import setup_logger
from detectron2.data import DatasetCatalog

from spine.classification import ClassificationEvaluator
from spine.config import add_spine_config
from spine.classification import setup_cls_data_catalog, add_classifier_config


def setup(args):
    cfg = get_cfg()
    add_spine_config(cfg)
    # add_sparsercnn_config(cfg)
    # add_efficientdet_config(cfg)
    add_classifier_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.OUTPUT_DIR = args.output_dir
    
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, name="spine")
    
    return cfg


def i_sigmoid(x):
    """
    inverted sigmoid
    """
    # clip for stability
    x = np.clip(x, 1e-6, 1-1e-6)
    return np.log(x / (1.-x))


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def ensemble_prediction(path_list):
    """
    load predictions saved by ClassificationEvaluator
    ensemble by average logit instead of probability
    """
    
    results = defaultdict(list)
    for path in path_list:
        result = torch.load(path)
        for img_id, pred in zip(result["image_ids"], result["preds"]):
            results[img_id].append(i_sigmoid(pred)) # shape (M,)
            
    for k, v in results.items():
        results[k] = sigmoid(np.stack(v).mean(axis=0))
        
    return results


def setup_evaluator(evaluator, predictions, dataset_dict):
    """
    setup prediction and groundtruth for evaluator
    args:
        evaluator: classification evaluator
        predictions: dict[<<img_id>>: array of shape (M,)] where M is number of labels
    """
    assert len(predictions) == len(dataset_dict)
    preds_list = []
    img_ids = []
    gts_list = []
    
    gts_dict = {x["image_id"]: x["classes"] for x in dataset_dict}
    for img_id, pred in predictions.items():
        img_ids.append(img_id)
        preds_list.append(pred[np.newaxis])
        gts_list.append(gts_dict[img_id][np.newaxis])
    
    evaluator.reset()
    evaluator._preds = preds_list
    evaluator._img_ids = img_ids
    evaluator._gts = gts_list
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ensemble classification predictions "
    )
    parser.add_argument("--output-dir", required=True, help="output directory")
    parser.add_argument("--dataset", help="name of the dataset", default="spine_cls_val")
    parser.add_argument("--config-file",type=str, required=True)
    args = parser.parse_args()
    cfg = setup(args)
    setup_cls_data_catalog(cfg)

    evaluator = ClassificationEvaluator(cfg, args.dataset, args.output_dir)
    
    predictions = ensemble_prediction([
        "./outputs/densenet121/inference/classification_predictions.pth",
        "./outputs/densenet169/inference/classification_predictions.pth",
        "./outputs/densenet201/inference/classification_predictions.pth",
    ])
    setup_evaluator(evaluator, predictions, DatasetCatalog.get(args.dataset))
    
    evaluator.evaluate()