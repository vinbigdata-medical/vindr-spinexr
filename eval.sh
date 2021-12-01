export PYTHONPATH=$(pwd)

CUDA_VISIBLE_DEVICES=0 python spine/train_net.py --num-gpus 1 \
  --config-file ./spine/configs/sparsercnn.yaml \
  --eval-only \
  MODEL.WEIGHTS ./outputs/sparsercnn/model_final.pth \
  MODEL.SparseRCNN.NMS_THRESH 0.7 \

# CUDA_VISIBLE_DEVICES=0 python spine/train_net.py --num-gpus 1 \
#   --config-file ./spine/configs/densenet121.yaml \
#   --eval-only \
#   MODEL.WEIGHTS ./outputs/densenet121/model_final.pth \

