export PYTHONPATH=$(pwd):$PYTHONPATH
python spine/ensemble_classifier.py \
  --config-file ./spine/configs/densenet121_dgx3.yaml \
  --output-dir ./outputs/ensemble_densenet