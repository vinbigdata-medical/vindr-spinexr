export PYTHONPATH=$(pwd)
CUDA_VISIBLE_DEVICES=0 python spine/train_net.py --num-gpus 1 \
  --config-file ./spine/configs/sparsercnn.yaml \
  # --resume

# CUDA_VISIBLE_DEVICES=0 python spine/train_net.py --num-gpus 1 \
#   --config-file ./spine/configs/densenet121.yaml \

# CUDA_VISIBLE_DEVICES=0 python spine/train_net.py --num-gpus 1 \
#   --config-file ./spine/configs/densenet169.yaml \

# CUDA_VISIBLE_DEVICES=0 python spine/train_net.py --num-gpus 1 \
#   --config-file ./spine/configs/densenet201.yaml \






