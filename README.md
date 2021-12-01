[<div style="text-align:center"><img src=".github/logo-VinBigData-2020-ngang-blue.png" width="300"></div>](https://vindr.ai/)

# VinDr-SpineXR: A deep learning framework forspinal lesions detection and classification from radiographs
<!-- <div align="center"> -->

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<!-- </div> -->

# Paper 
 [VinDr-SpineXR: A deep learning framework forspinal lesions detection and classification from radiographs](https://link.springer.com/chapter/10.1007/978-3-030-87240-3_28) \
 \[[preprint](https://arxiv.org/abs/2106.12930), [poster](https://drive.google.com/file/d/1IiLivi_VQ91W4R7RJuEz2Fw7E_df8evE/view?usp=sharing)\]
# Installation
To install in Docker container, see [Docker instruction](docker/README.md)

To install via Pip, run 
```bash
pip install -r ./docker/requirements.txt
```


# Data Preparation
The dataset can be downloaded via our project on Physionet. Subsequently, images need to be converted into PNG format before training. For more detail see [data preparation instructions](data/README.md)

# Training
For each experiment outputs will be stored in `outputs/--exp-name--`. Refer to [train.sh](train.sh) for more example scripts.
To train Sparse R-CNN, COCO pre-trained checkpoint has to be manually downloaded (see [here](data/README.md) for detail).
It takes about one day to train Sparse R-CNN on a V100 32GiB.

```bash
CUDA_VISIBLE_DEVICES=0 python spine/train_net.py --num-gpus 1 \
  --config-file ./spine/configs/sparsercnn.yaml \
```

# Evaluation
Refer to [eval.sh](eval.sh) for more example scripts.
```bash
CUDA_VISIBLE_DEVICES=0 python spine/train_net.py --num-gpus 1 \
  --config-file ./spine/configs/sparsercnn.yaml \
  --eval-only \
  MODEL.WEIGHTS ./outputs/sparsercnn/model_final.pth \
  MODEL.SparseRCNN.NMS_THRESH 0.7 \
```
# Visualize
```bash
python spine/visualize_json_results.py \
    --input "./outputs/sparsercnn/inference/coco_instances_results.json" \
    --output "./outputs/sparsercnn/inference/visual" \
    --config-file "./spine/configs/sparsercnn.yaml" \
    --dataset "spine_test" \
    --conf-threshold 0.25
```
# License
This source code in released under [Apache 2.0 License](LICENSE).


# Acknowledgment
This implementation is based on [Detection2](https://github.com/facebookresearch/detectron2) codebase.\
Thanks to [Benjin Zhu](https://github.com/poodarchu) for [numpy implementation](https://github.com/poodarchu/learn_aug_for_object_detection.numpy/tree/add-license-1) of [AutoAugment for object detection](https://link.springer.com/chapter/10.1007%2F978-3-030-58583-9_34). \
Thanks to [Ross Wightman](https://github.com/rwightman) for [his implementation](https://github.com/rwightman/efficientdet-pytorch) of [EfficientDet](https://openaccess.thecvf.com/content_CVPR_2020/html/Tan_EfficientDet_Scalable_and_Efficient_Object_Detection_CVPR_2020_paper.html).\
Thanks to authors of [Sparse R-CNN](https://github.com/PeizeSun/SparseR-CNN).


# Citing
If you use the VinDr-SpineXR dataset in your research or wish to refer to the baseline results published in our paper, please use the following BibTeX for citation:

```BibTeX
@InProceedings{nguyen2021vindr,
    author="Nguyen, Hieu T.
        and Pham, Hieu H.
        and Nguyen, Nghia T.
        and Nguyen, Ha Q.
        and Huynh, Thang Q.
        and Dao, Minh
        and Vu, Van",
    editor="de Bruijne, Marleen
        and Cattin, Philippe C.
        and Cotin, St{\'e}phane
        and Padoy, Nicolas
        and Speidel, Stefanie
        and Zheng, Yefeng
        and Essert, Caroline",
    title="VinDr-SpineXR: A Deep Learning Framework for Spinal Lesions Detection and Classification from Radiographs",
    booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2021",
    year="2021",
    publisher="Springer International Publishing",
    address="Cham",
    pages="291--301",
    isbn="978-3-030-87240-3"
}
```
