echo "parse training dicom"
python spine/preprocess/parse_dicom.py \
  --dicom-folder "./data/train_images" \
  --output-file "./data/train_meta.csv" \
  --cpus 4 \
#   --debug

# echo "parse test dicom"
# python spine/preprocess/parse_dicom.py \
#   --dicom-folder "./data/test_images" \
#   --output-file "./data/test_meta.csv" \
#   --cpus 4 \
#   --debug
