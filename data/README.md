# Data preparation

## Data Downloading
The dataset can be downloaded via [our project on Physionet](https://physionet.org/content/vindr-spinexr/1.0.0/).
The dataset should be downloaded into this folder, data folder structure after downloading:
```bash
"project-home"/data
├── annotation
│   ├── train.json
|   └── test.json
├── train_images
│   ├── 00073745e02e69432c002b527c565151.dicom
│   ├── ...
|   └── fffa8adcc5e692cdb816051b6202870d.dicom
└── test_images
    ├── 004004095d8a302b1c0815ccb044c018.dicom
    ├── ...
    └── ff6a81f9fa386401ce11a0eb74e1f661.dicom
```



## Converting DICOM images to PNG format 


Run the [bash script](convert_dicom.sh). Note that you might need to change the number of CPU workers.
```bash
bash convert_dicom.sh
```

Data folder structure after converting:
```bash
"project-home"/data
├── annotation
├── train_images
├── test_images
├── train_pngs
│   ├── 00073745e02e69432c002b527c565151.png
│   ├── ...
|   └── fffa8adcc5e692cdb816051b6202870d.png
└── test_pngs
    ├── 004004095d8a302b1c0815ccb044c018.png
    ├── ...
    └── ff6a81f9fa386401ce11a0eb74e1f661.png
```

## Parse DICOM metadata
To read metadata of DICOM images, run the [parse_dicom.sh](parse_dicom.sh). Since Detectron2 requires image size for data augmentation, this step is essential.