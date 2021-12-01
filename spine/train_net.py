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


import os
import logging
from collections import OrderedDict

from detectron2.engine import (
    DefaultTrainer, 
    default_argument_parser, 
    default_setup, 
    launch,
)
from detectron2.evaluation import inference_on_dataset, verify_results, print_csv_format
from detectron2.data import build_detection_train_loader, build_detection_test_loader
import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger
from detectron2.utils import comm
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer

from spine.config import add_spine_config
from spine.dataset_dict import SpineDatasetFunction, setup_data_catalog
from spine.dataset_mapper import *
from spine.coco_evaluator import CustomCOCOEvaluator
from spine.sparsercnn import add_sparsercnn_config, SparseRCNN
from spine.classification import (
    ClassificationEvaluator, 
    build_classification_train_loader,
    build_classification_test_loader,
    add_classifier_config,
    setup_cls_data_catalog
)
import spine.classification


class Trainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DATA_MAPPER_REGISTRY.get(cfg.SPINE.DATA_MAPPER)(cfg, mode="train")
        if cfg.SPINE.TASK == "classification":
            return build_classification_train_loader(cfg, mapper=mapper)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = DATA_MAPPER_REGISTRY.get(cfg.SPINE.DATA_MAPPER)(cfg, mode="test")
        if cfg.SPINE.TASK == "classification":
            return build_classification_test_loader(cfg, mapper=mapper, dataset_name=dataset_name)
        return build_detection_test_loader(cfg, mapper=mapper, dataset_name=dataset_name)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_dir=None):
        if output_dir is None:
            output_dir = os.path.join(cfg.OUTPUT_DIR, "inference")
        if cfg.SPINE.TASK == "classification":
            return ClassificationEvaluator(dataset_name=dataset_name, output_dir=output_dir, cfg=cfg)
        evaluator = CustomCOCOEvaluator(dataset_name=dataset_name, output_dir=output_dir, cfg=cfg)

        return evaluator


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_spine_config(cfg)
    add_sparsercnn_config(cfg)
    add_classifier_config(cfg)
    
    cfg.merge_from_file(args.config_file)

    if cfg.SPINE.ARCH in ["retina_r50", "faster_rcnn_r50_fpn"]:
        arch_cfg = {
            "retina_r50": "COCO-Detection/retinanet_R_50_FPN_1x.yaml",
            "faster_rcnn_r50_fpn": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        }[cfg.SPINE.ARCH]
        cfg.merge_from_file(model_zoo.get_config_file(arch_cfg))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(arch_cfg)

    
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.OUTPUT_DIR = os.path.join(
        cfg.OUTPUT_DIR, 
        os.path.split(args.config_file)[-1].replace(".yaml", ""))
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="spine")
    return cfg


def main(args):
    cfg = setup(args)
    if cfg.SPINE.TASK == "detection":
        setup_data_catalog(cfg)
    elif cfg.SPINE.TASK == "classification":
        setup_cls_data_catalog(cfg)
    
    # eval
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = OrderedDict()
        res.update(Trainer.test(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
            print(res)
        return res
    
    # train
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
