echo "convert training images to png"
python spine/preprocess/dicom2png.py \
  --input-dir "./data/train_images" \
  --output-dir "./data/train_pngs" \
  --cpus 4 \
  --log-file "./data/convert_train_log.txt" \
  --out-file "./data/convert_train_results.csv" \
#   --debug

# echo "convert test ima/dicom2png.py \
#   --input-dir "./data/tges to png"
# python spine/preprocessest_images" \
#   --output-dir "./data/test_pngs" \
#   --cpus 4 \
#   --log-file "./data/convert_test_log.txt" \
#   --out-file "./data/convert_test_results.csv" \
#   --debug