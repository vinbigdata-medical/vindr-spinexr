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


import torch
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.engine import default_setup, default_argument_parser

from spine.dataset_dict import setup_data_catalog
from spine.config import add_spine_config
from spine.sparsercnn import add_sparsercnn_config
from spine.coco_evaluator import CustomCOCOEvaluator


def setup(args):
    cfg = get_cfg()
    add_spine_config(cfg)
    add_sparsercnn_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.OUTPUT_DIR = args.output_dir
    
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, name="spine")
    
    return cfg



def combine_results(d_results, cls_results, abnormal_threshold, keep_threshold):
    """
    For the images which are classified as normal, 
    keep boxes which have higher confident than a threshold 
    """
    for img_pred in d_results:
        abnormal_prob = cls_results[img_pred["image_id"]]
        normal = abnormal_prob < abnormal_threshold
        if normal:
            kept_instances = []
            for instance in img_pred["instances"]:
                if instance["score"] >= keep_threshold:
                    kept_instances.append(instance)
            img_pred["instances"] = kept_instances
    

def read_detection_results(d_path):
    return torch.load(d_path)


def read_cls_results(cls_path):
    results = torch.load(cls_path)
    
    results_dict = {img_id: float(pred) # assume there is only 1 label
        for img_id, pred in zip(results["image_ids"], results["preds"])}
    return results_dict


def setup_evaluator(evaluator, predictions):
    """
    setup prediction and groundtruth for evaluator
    args:
        evaluator: Custom coco evaluator
        
    """
    evaluator.reset()
    evaluator._predictions = predictions
    
    
if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--cls-results", required=True, help="JSON file produced by the model")
    parser.add_argument("--det-results", required=True, help="output directory")
    parser.add_argument("--abnormal-threshold", required=True, type=float, 
        help="confidence threshold for classification")
    parser.add_argument("--keep-threshold", default=0.5, type=float, 
        help="confidence threshold to keep boxes")
    parser.add_argument("--dataset", help="name of the dataset", default="spine_test")
    parser.add_argument("--output-dir", required=True, help="output directory")
    args = parser.parse_args()
    cfg = setup(args)
    setup_data_catalog(cfg)
    d_results = read_detection_results(args.det_results)
    cls_results = read_cls_results(args.cls_results)
    combine_results(d_results, cls_results, args.abnormal_threshold, args.keep_threshold)
    evaluator = CustomCOCOEvaluator(cfg, args.dataset, output_dir=args.output_dir)
    setup_evaluator(evaluator, d_results)
    evaluator.evaluate()
