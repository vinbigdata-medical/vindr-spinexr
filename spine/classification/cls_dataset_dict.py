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
import numpy as np
import pandas as pd
from detectron2.data import DatasetCatalog, MetadataCatalog


def setup_cls_data_catalog(cfg):
    """
    register datasetcatalog and metadata_catalog
    """
    # register dataset_catalog
    dataset_dict_fn = SpineClsDatasetFunction(cfg, "train")
    DatasetCatalog.register("spine_cls_train", dataset_dict_fn)

    dataset_dict_fn = SpineClsDatasetFunction(cfg, "test")
    DatasetCatalog.register("spine_cls_test", dataset_dict_fn)

    # register metadata_catalog
    classes = cfg.MODEL.CLASSIFIER.CLASSES
    MetadataCatalog.get("spine_cls_train").thing_classes = classes
    MetadataCatalog.get("spine_cls_test").thing_classes = classes
    
    print(
        MetadataCatalog.get("spine_cls_train"), "\n",
        MetadataCatalog.get("spine_cls_test")
    )


class SpineClsDatasetFunction:
    def __init__(self, cfg, mode="train"):

        assert mode in ["train", "test"]
        self.mode = mode
        self.cfg = cfg

    def __call__(self):
        """
        return list[dict]
        """
        cfg = self.cfg
        disease_classes = cfg.MODEL.CLASSIFIER.CLASSES

        data_folder = cfg.SPINE.TRAIN_DATA_FOLDER if self.mode == "train" else cfg.SPINE.TEST_DATA_FOLDER
        metadata = cfg.SPINE.TRAIN_METADATA if self.mode == "train" else cfg.SPINE.TEST_METADATA
        metadata = pd.read_csv(metadata)
        metadata = metadata[["image_id", "image_height", "image_width"]]
        metadata = metadata.set_index("image_id")
        metadata = metadata.to_dict(orient="index")
        annotations = cfg.SPINE.TRAIN_ANNOTATION if self.mode == "train" else cfg.SPINE.TEST_ANNOTATION
        annotations = pd.read_csv(annotations)
        
        abnormal_only = "Abnormal" in disease_classes
        if abnormal_only:
            assert len(disease_classes) == 1
            
        dataset_dict = []
        for image_id, rows in annotations.groupby("image_id"):
            instance_dict = {}
            instance_dict["file_name"] = os.path.join(data_folder, f"{image_id}.png")
            instance_dict["image_id"] = image_id
            instance_dict["height"] = metadata[image_id]["image_height"]
            instance_dict["width"] = metadata[image_id]["image_width"]
            classes = rows["lesion_type"].unique().tolist()
            if abnormal_only:
                label = 0. if "No finding" in classes else 1.
                labels = np.array([label])
            else:
                labels = np.array([0.0]*len(disease_classes))
                for label in classes:
                    if (label not in disease_classes) and "Other disease" in disease_classes:
                        label = "Other disease"
                    labels[disease_classes.index(label)] = 1.
            
            instance_dict["classes"] = labels
            dataset_dict.append(instance_dict)
        
        return dataset_dict