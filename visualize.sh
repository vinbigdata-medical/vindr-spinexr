
export PYTHONPATH=$(pwd) 
python spine/visualize_json_results.py \
    --input "./outputs/sparsercnn/inference/coco_instances_results.json" \
    --output "./outputs/sparsercnn/inference/visual" \
    --config-file "./spine/configs/sparsercnn.yaml" \
    --dataset "spine_test" \
    --conf-threshold 0.25


