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
import itertools
import logging
import math
from bisect import bisect_right
import torch
from collections import OrderedDict, defaultdict
from tabulate import tabulate
from sklearn.metrics import roc_curve, roc_auc_score, f1_score
import matplotlib.pyplot as plt

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils.file_io import PathManager
import detectron2.utils.comm as comm


class ClassificationEvaluator(DatasetEvaluator):
    def __init__(self, cfg, dataset_name, output_dir, distributed=True):
        # self.cfg = cfg
        dataset_dicts = DatasetCatalog.get(dataset_name)
        self._dataset_dicts = {x["image_id"]: x for x in dataset_dicts}
        self._metadata = MetadataCatalog.get(dataset_name)
        self._distributed = distributed
        self._logger = logging.getLogger(__name__)
        self._bootstrap = cfg.CLASSIFICATION.BOOTSTRAP
        self._output_dir = output_dir
        if self._bootstrap:
            self._bootstrap_samples = cfg.CLASSIFICATION.BOOTSTRAP_SAMPLES
            self._bootstrap_ci = cfg.CLASSIFICATION.BOOTSTRAP_CI
            self._seed = cfg.SEED

    def process(self, inputs, outputs):
        """
        cache inputs and outputs
        
        args:
            inputs(inputs of model): list of input dict
            outputs(outputs of model): tensor of shape (N,M)
        """
        assert len(inputs) == len(outputs)

        self._img_ids.extend([x["image_id"] for x in inputs])
        self._preds.append(outputs.detach().cpu().numpy())
        
    def reset(self):
        self._preds = []
        self._img_ids = []
    
    def evaluate(self, thresholds=None):
        """
        run evaluation
        """
        self._logger.info("Evaluating using ClassificationEvaluator")

        if self._distributed:
            print(f"rank {comm.get_rank()}: {len(self._preds)}, img_id {len(set(self._img_ids))}")
            comm.synchronize()
            probs = comm.gather(self._preds, dst=0)
            probs = list(itertools.chain(*probs))

            img_ids = comm.gather(self._img_ids, dst=0)
            img_ids = list(itertools.chain(*img_ids))

            if not comm.is_main_process():
                return {}

            print(f"main:{len(probs)}, img_id {len(set(img_ids))}")
        else:
            probs = self._preds
            img_ids = self._img_ids

        all_preds_dict = {}
        for img_id, pred in zip(img_ids, probs):
            all_preds_dict[img_id] = pred
        
        # cummulate prediction
        all_preds, all_gts, img_ids = [], [], []
        for img_id, pred in all_preds_dict.items():
            all_preds.append(pred)
            all_gts.append(self._dataset_dicts[img_id]["classes"])
            img_ids.append(img_id)
        all_preds = np.stack(all_preds, axis=0)
        all_gts = np.stack(all_gts, axis=0)
        
        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "classification_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save({"image_ids": img_ids, "preds": all_preds}, f)
        
        results_dict = OrderedDict()

        if thresholds is None: 
            _thresholds = {k: None for k in self._metadata.thing_classes}
        else: 
            _thresholds = thresholds
        results_dict["Classification"] = self._compute_metrics(all_preds, all_gts, _thresholds)
        if self._bootstrap:
            if thresholds is None: 
                thresholds = {}
                for k,v in results_dict["Classification"].items():
                    if "threshold" in k:
                        thresholds[k.replace("_threshold", "")] = v
            results_dict["Classification bootstrap"] = \
                self._compute_bootstrap_metrics(all_preds, all_gts, thresholds)
        return results_dict

    def _compute_metrics(self, preds, gts, thresholds):
        
        headers = ["Class", "AUC", "F1", "Sensitivity", "Specificity", "Precision" ,"threshold"]
        results = []
        for idx, _class in enumerate(self._metadata.thing_classes):
            result_per_class = self._compute_metrics_single_class(
                preds=preds[:,idx], gts=gts[:,idx], save_roc_outputs=True, 
                threshold=thresholds[_class],
            )
            results.append([_class] + [result_per_class[x] for x in headers[1:]])
        
        table = tabulate(
            results,
            tablefmt="pipe",
            floatfmt=".4f",
            headers=headers,
            numalign="left",
        )
        self._logger.info("Classification results: \n" + table)
        
        results_dict = OrderedDict()
        for class_results in results:
            for name , value in zip(headers[1:], class_results[1:]):
                results_dict[f"{class_results[0]}_{name}"] = value
        return results_dict
    
    def _compute_metrics_single_class(self, preds, gts, *, save_roc_outputs=False, threshold=None):   
        """
        compute AUC and 
        f1, sensitivity, specificity base on optimal threshold for 
        Youden's J statistic (sensitivity + specificity -1)
        
        args: 
            preds(np.array): probs of shape (N)
            gts(np.array): gt of shape (N)
            threshold: float in [0., 1.] or None, 
                if None, it is chosen using Youden's J statistic
        """
        
        # compute auc
        results = {"AUC": roc_auc_score(gts, preds, average=None)}
        # draw roc curve and get optimal threshold for J index
        fpr, tpr, thresholds = roc_curve(gts, preds)
        if save_roc_outputs:
            file_path = os.path.join(self._output_dir, "fpr_tpr_thresh.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save({"fpr": fpr, "tpr": tpr, "threshold": thresholds}, f)
            fig_path = os.path.join(self._output_dir, "roc_curve.png")
            plt.figure(figsize=(5,5))
            plt.plot(fpr, tpr,)
            plt.xlabel('1 - Specificity')
            plt.ylabel('Sensitivity')
            plt.savefig(fig_path)
        if threshold is not None:
            thresh_i = len(thresholds) - 1 - bisect_right(np.flip(thresholds), threshold)
        else:
            J = tpr - fpr
            thresh_i = np.argmax(J)
            threshold = thresholds[thresh_i]

        tpr = tpr[thresh_i]
        fpr = fpr[thresh_i]
        results["threshold"] = threshold
        results["Sensitivity"] = tpr
        results["Specificity"] = 1 - fpr
        preds_cls = (preds>=threshold).astype(np.int)
        results["Precision"] = tpr * float(gts.sum()) / float(preds_cls.sum())
        results["F1"] = f1_score(gts, preds_cls)
        if save_roc_outputs:
            plt.plot(fpr, tpr, marker='.', color="r", 
                label="Optimal cutoff point for Youden's J statistic")
            plt.legend(loc="lower right")
            plt.grid(linestyle="--")
            plt.savefig(fig_path)
        return results

    def _compute_bootstrap_metrics(self, preds, gts, thresholds):
        """
        compute metrics with bootstraping: AUC, sensitivity, specificity, f1

        For the three latter metrics, the decision threshold is chosen to optimize 
        Jouden's J statistic based on the mean ROC curve

        args: 
            preds(np.array): probs of shape (N,M)
            gts(np.array): gt of shape (N,M)
        
        NOTE: each call run from the same numpy random state
        """
        print(thresholds)
        random_state = np.random.RandomState(self._seed)
        headers = ["Class", "AUC", "F1", "Sensitivity", "Specificity", "Precision", "threshold"]
        N = len(preds)
        all_stats = {x: defaultdict(list) for x in headers[1:]}
        # run bootstrap
        b_count = 0
        while b_count < self._bootstrap_samples:

            rand_ids = random_state.randint(N, size=N)
            sampled_pred, sampled_gts = preds[rand_ids], gts[rand_ids]
            # check if there is any label has only 1 class
            is_skip = False
            for _class_id in range(sampled_gts.shape[1]):
                if len(np.unique(sampled_gts[:,_class_id])) == 1:
                    is_skip = True
                    break
            if is_skip: 
                continue

            for idx, _class in enumerate(self._metadata.thing_classes):
                pred_per_class, gts_per_class = sampled_pred[:,idx], sampled_gts[:,idx]
                results = self._compute_metrics_single_class(
                    pred_per_class, gts_per_class, threshold=thresholds[_class])
                for name in headers[1:]:
                    all_stats[name][_class].append(results[name])

            b_count += 1

        # print results in table
        def fmt_float(x):
            # assert x < 1.
            return f"{x:.4f}"

        results_table = []
        for idx, _class in enumerate(self._metadata.thing_classes):
            row = [_class]
            for metric in headers[1:-1]:
                mean, lb, ub = self._get_mean_ci(np.array(all_stats[metric][_class]))
                mean, lb, ub = float(mean), float(lb), float(ub)
                row.append(f"{fmt_float(mean)}({fmt_float(lb)},{fmt_float(ub)})")
            row.append(f"{fmt_float(all_stats['threshold'][_class][0])}")
            results_table.append(row)
        
        table = tabulate(
            results_table,
            tablefmt="pipe",
            floatfmt=".4f",
            headers=headers,
            numalign="left",
        )
        self._logger.info("Classification bootstrap results: \n" + table)
        
        results_dict = OrderedDict()
        for class_results in results_table:
            for name , value in zip(headers[1:], class_results[1:]):
                results_dict[f"{class_results[0]}_b_mean_{name}"] = float(value[:6])
        return results_dict
    
    def _get_mean_ci(self, values):
        """
        compute mean, confident interval of a variable or multiple variables

        values: np.array of shape (N,M) or (N,) 
            where N is number of samples, M is number of variables
        return mean, lb, ub

        NOTE: sort inplace the given values
        """
        if len(values.shape) == 1:
            values = values[:, np.newaxis]
        N = len(values)
        tail = (1. - self._bootstrap_ci) / 2.
        bound_id = math.floor(tail*N)

        values.sort(axis=0)
        mean = values.mean(axis=0)
        lb = values[bound_id]
        ub = values[-bound_id]

        return mean, lb, ub


if __name__ == "__main__":
    class CFG:
        pass

    cfg = CFG()
    cfg.CLASSIFICATION = CFG()
    cfg.CLASSIFICATION.BOOTSTRAP_SAMPLES = 1000
    cfg.CLASSIFICATION.BOOTSTRAP_CI = 0.95
    cfg.CLASSIFICATION.BOOTSTRAP = True

    values = np.arange(2000)
    np.random.shuffle(values)

    eval = ClassificationEvaluator(cfg, "", "")
    print(eval._get_mean_ci(values))