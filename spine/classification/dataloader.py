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


import itertools
import logging
import numpy as np
import torch

from termcolor import colored
from tabulate import tabulate

from detectron2.config import configurable
from detectron2.data import (
    DatasetCatalog, 
    MetadataCatalog, 
    build_batch_data_loader,
)
from detectron2.data.detection_utils import check_metadata_consistency
from detectron2.data.samplers import (
    InferenceSampler,
    TrainingSampler,
)
from detectron2.data import DatasetFromList, MapDataset
from detectron2.data.build import trivial_batch_collator
from detectron2.utils.logger import log_first_n


def print_instances_class_histogram(dataset_dicts, class_names):
    """
    similar to `print_instances_class_histogram` at
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/build.py, 

    adapted for classification annotation in one-hot format
    
    Args:
        dataset_dicts (list[dict]): list of dataset dicts.
        class_names (list[str]): list of class names (zero-indexed).
    """
    num_classes = len(class_names)
    hist_bins = np.arange(num_classes + 1)
    histogram = np.zeros((num_classes,), dtype=np.float)
    for entry in dataset_dicts:
        classes = entry["classes"]
        
        assert classes.shape[0] == num_classes, \
            f"Got an invalid classes length {classes}, expect {num_classes}"
        assert classes.min() in [0., 1.], f"Got an invalid classes values ={classes}, expect 0 1"
        assert classes.max() in [0., 1.], f"Got an invalid classes values ={classes}, expect 0 1"
        
        histogram += classes 

    N_COLS = min(6, len(class_names) * 2)

    def short_name(x):
        # make long class names shorter. useful for lvis
        if len(x) > 13:
            return x[:11] + ".."
        return x

    data = list(
        itertools.chain(*[[short_name(class_names[i]), int(v)] for i, v in enumerate(histogram)])
    )
    total_num_instances = sum(data[1::2])
    data.extend([None] * (N_COLS - (len(data) % N_COLS)))
    if num_classes > 1:
        data.extend(["total", total_num_instances])
    data = itertools.zip_longest(*[data[i::N_COLS] for i in range(N_COLS)])
    table = tabulate(
        data,
        headers=["category", "#instances"] * (N_COLS // 2),
        tablefmt="pipe",
        numalign="left",
        stralign="center",
    )
    log_first_n(
        logging.INFO,
        "Distribution of instances among all {} classes:\n".format(num_classes)
        + colored(table, "cyan"),
        key="message",
    )


def get_classidication_dataset_dicts(dataset_names):
    """
    Load and join classification dataset dicts

    Args:
        dataset_names (str or list[str]): a dataset name or a list of dataset names


    Returns:
        list[dict]: a list of dicts following the standard dataset dict format.
    """
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    assert len(dataset_names)
    dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in dataset_names]
    for dataset_name, dicts in zip(dataset_names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)


    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))

    has_labels= "classes" in dataset_dicts[0]

    if has_labels:
        try:
            class_names = MetadataCatalog.get(dataset_names[0]).thing_classes
            check_metadata_consistency("thing_classes", dataset_names)
            print_instances_class_histogram(dataset_dicts, class_names)
        except AttributeError:  # class names are not available for this dataset
            pass

    assert len(dataset_dicts), "No valid data found in {}.".format(",".join(dataset_names))
    return dataset_dicts


def _train_loader_from_config(cfg, mapper, *, dataset=None, sampler=None):
    if dataset is None:
        dataset = get_classidication_dataset_dicts(cfg.DATASETS.TRAIN)

    if sampler is None:
        sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
        logger = logging.getLogger(__name__)
        logger.info("Using training sampler {}".format(sampler_name))
        if sampler_name == "TrainingSampler":
            sampler = TrainingSampler(len(dataset))
        elif sampler_name == "RepeatFactorTrainingSampler":
            raise NotImplementedError()
        else:
            raise ValueError("Unknown training sampler: {}".format(sampler_name))

    return {
        "dataset": dataset,
        "sampler": sampler,
        "mapper": mapper,
        "total_batch_size": cfg.SOLVER.IMS_PER_BATCH,
        "aspect_ratio_grouping": cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
    }


@configurable(from_config=_train_loader_from_config)
def build_classification_train_loader(
    dataset, *, mapper, sampler=None, total_batch_size, aspect_ratio_grouping=True, num_workers=0
):
    """
    similar to `build_detection_train_loader` at 
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/build.py
    """
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    if sampler is None:
        sampler = TrainingSampler(len(dataset))
    assert isinstance(sampler, torch.utils.data.sampler.Sampler)
    return build_batch_data_loader(
        dataset,
        sampler,
        total_batch_size,
        aspect_ratio_grouping=aspect_ratio_grouping,
        num_workers=num_workers,
    )


def _test_loader_from_config(cfg, dataset_name, mapper):
    dataset = get_classidication_dataset_dicts([dataset_name])
    return {"dataset": dataset, "mapper": mapper, "num_workers": cfg.DATALOADER.NUM_WORKERS}


@configurable(from_config=_test_loader_from_config)
def build_classification_test_loader(dataset, *, mapper, num_workers=0):
    """
    similar to `build_detection_test_loader` at 
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/build.py

    """
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    sampler = InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader




