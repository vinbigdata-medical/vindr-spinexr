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

import pydicom
import numpy as np
import cv2
import pandas as pd
import argparse
from pathlib import Path
from joblib import Parallel, delayed
from pydicom.pixel_data_handlers.util import apply_voi_lut


def log_(msg, log_file):
    print(str(msg))
    with open(log_file, 'a') as f:
        f.write(str(msg) + '\n')


def process_dicom_image(dicom_file, outdir, log_file):
    imageID = dicom_file.stem

    voi_lut = True
    png = convert2(str(dicom_file))
    if png is None:
        png = convert1(dicom_file)
        voi_lut = False
        log_(f"{str(dicom_file)} cannot apply vui lut", log_file)

    has_png = True
    if png is not None:
        outfile = outdir/f'{imageID}.png'
        cv2.imwrite(outfile.as_posix(), png)
    else:
        log_(f"{str(dicom_file)} cannot convert", log_file)
        has_png = False

    return {'image_id': imageID, "has_png": has_png, "voi_lut": voi_lut}


def convert1(dcm_file_path):
    """
    convert by equalhist
    """
    dcm_file = pydicom.dcmread(dcm_file_path)
    if dcm_file.BitsStored in (10,12):
            dcm_file.BitsStored = 16
    try:
        rescaled_image = cv2.convertScaleAbs(dcm_file.pixel_array,
                                            alpha=(255.0/dcm_file.pixel_array.max()))
    except Exception as e:
        print(e)
        return None
    # Correct image inversion.
    if dcm_file.PhotometricInterpretation == "MONOCHROME1":
        rescaled_image = cv2.bitwise_not(rescaled_image)
    adjusted_image = cv2.equalizeHist(rescaled_image)
    # adjusted_image = rescaled_image
    return adjusted_image


def convert2(dcm_file_path, voi_lut=True):
    """
    convert by voi_lui function from pydicom
    """
    dicom = pydicom.read_file(dcm_file_path)
    try:
        if voi_lut:
            data = apply_voi_lut(dicom.pixel_array, dicom)
        else:
            data = dicom.pixel_array
    except Exception as e:
        print(e)
        return None
    # Correct image inversion.
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)

    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True, type=str)
    parser.add_argument('--output-dir', required=True, type=str)
    parser.add_argument('--cpus', type=int, default=4, help='Number of workers')
    parser.add_argument('--out-file', required=True, type=str)
    parser.add_argument('--log-file', required=True, type=str, help='path to log file')
    parser.add_argument('--debug', action='store_true', help='Run on subset of dataset to debug')

    args = parser.parse_args()
    indir = Path(args.input_dir)
    outdir = Path(args.output_dir)
    log_file = args.log_file
    out_file = args.out_file
    outdir.mkdir(exist_ok=True, parents=True)
    print("conver dicom from", str(indir))
    dicom_files = indir.glob('*dicom')

    dicom_files = list(dicom_files)
    import numpy as np
    np.random.shuffle(dicom_files)
    if args.debug:
        log_file = log_file.replace(".txt", ".debug.txt")
        out_file = out_file.replace(".csv", ".debug.csv")
        def early_stop(gen, stop=20):
            for i, e in enumerate(gen):
                if i == stop: break
                yield e
        dicom_files = early_stop(dicom_files, 20)
        args.cpus = 1

    executor = Parallel(n_jobs=args.cpus, backend='multiprocessing', prefer='processes', verbose=1)
    do = delayed(process_dicom_image)
    tasks = (do(f, outdir, log_file) for f in dicom_files)
    results = executor(tasks)

    results = pd.DataFrame(results)
    results.to_csv(out_file, index=False)
