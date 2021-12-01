export PYTHONPATH=$(pwd):$PYTHONPATH


python spine/eval_overall.py \
  --config-file ./spine/configs/sparsercnn.yaml \
  --output-dir ./outputs/overall_pipeline \
  --cls-results "./outputs/ensemble_densenet/classification_predictions.pth" \
  --det-results "./outputs/sparsercnn/inference/instances_predictions.pth" \
  --abnormal-threshold 0.28 \ # get from evaluation result of prediction ensemble

