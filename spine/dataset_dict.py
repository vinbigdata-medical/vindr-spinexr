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
import copy
import logging
import pandas as pd
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog


def setup_data_catalog(cfg):
    """
    register datasetcatalog and metadata_catalog
    """
    # register dataset_catalog
    if cfg.SPINE.DATASET_DICT_FN == "SpineDatasetFunction":
        DatasetFunction = SpineDatasetFunction

    else:
        raise ValueError("invalid dataset_dict_function name")
    dataset_dict_fn = DatasetFunction(cfg, "train")
    DatasetCatalog.register("spine_train", dataset_dict_fn)

    dataset_dict_fn = DatasetFunction(cfg, "test")
    DatasetCatalog.register("spine_test", dataset_dict_fn)

    # register metadata_catalog
    finding_classes = cfg.SPINE.FINDINGS
    MetadataCatalog.get("spine_train").thing_classes = finding_classes
    MetadataCatalog.get("spine_test").thing_classes = finding_classes
    
    id_map = {i: i for i in range(len(finding_classes))}
    MetadataCatalog.get("spine_train").thing_dataset_id_to_contiguous_id = id_map
    MetadataCatalog.get("spine_test").thing_dataset_id_to_contiguous_id = id_map
    print(
        MetadataCatalog.get("spine_train"), "\n",
        MetadataCatalog.get("spine_test")
    )


class SpineDatasetFunction:
    def __init__(self, cfg, mode="train"):

        assert mode in ["train", "test"]
        self.mode = mode
        self.cfg = cfg

    def __call__(self):
        """
        parse label from csv
        return list[dict]
        """
        
        logger = logging.getLogger(__name__)
        cfg = self.cfg
        finding_classes = cfg.SPINE.FINDINGS

        data_folder = cfg.SPINE.TRAIN_DATA_FOLDER if self.mode == "train" else cfg.SPINE.TEST_DATA_FOLDER
        metadata = cfg.SPINE.TRAIN_METADATA if self.mode == "train" else cfg.SPINE.TEST_METADATA
        metadata = pd.read_csv(metadata)
        metadata = metadata[["image_id", "image_height", "image_width"]]
        metadata = metadata.set_index("image_id")
        metadata = metadata.to_dict(orient="index")
        annotations = cfg.SPINE.TRAIN_ANNOTATION if self.mode == "train" else cfg.SPINE.TEST_ANNOTATION
        annotations = pd.read_csv(annotations)
        
        dataset_dict = []
        for image_id, rows in annotations.groupby("image_id"):
            instance_dict = {}
            instance_dict["file_name"] = os.path.join(data_folder, f"{image_id}.png")
            instance_dict["image_id"] = image_id
            instance_dict["height"] = metadata[image_id]["image_height"]
            instance_dict["width"] = metadata[image_id]["image_width"]

            objs = []
            for _, row in rows.iterrows():
                lesion_type = row["lesion_type"]
                if lesion_type not in finding_classes:
                    continue
                box = (row["xmin"], row["ymin"], row["xmax"], row["ymax"])
                # remove empty box
                if (box[2]-box[0]) < 1e-3 or (box[3]-box[1]) < 1e-3:
                    logger.info(f"remove empty box in image {image_id}, study {row['study_id']}: {box}; {lesion_type}")
                    continue

                objs.append({
                    "bbox": copy.deepcopy(box),
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": finding_classes.index(lesion_type),
                    "iscrowd": 0,
                })
            
            instance_dict["annotations"] = objs
            dataset_dict.append(instance_dict)
        
        return dataset_dict
