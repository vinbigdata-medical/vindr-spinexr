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


import os
from pathlib import Path
import pandas as pd
import collections
import pydicom
import argparse
from tqdm import tqdm
from joblib import Parallel, delayed

def log_failed(msg):
    with open(f'failed.txt', 'a') as f:
        f.write(str(msg) + '\n')


def process_dicom(dicom_file):
    try:
        data = pydicom.read_file(dicom_file)
    except Exception as e:
        print(dicom_file, e)
        log_failed(dicom_file)
        return {}
    # ImageType
    imageType = data.get("ImageType", "") # ORIGINAL, DERIVED/SECONDARY 
    imageType = filter(lambda e: e != '', imageType)
    imageType = '/'.join(imageType)

    result = {
        'patient_id': data.get('PatientID', ''),
        'image_id': data.get('SOPInstanceUID', ''),
        'series_id': data.get('SeriesInstanceUID', ''),
        'study_id': data.get('StudyInstanceUID', ''),
        'image_type': imageType,
        'image_height': data.get("Rows", 0),
        'image_width': data.get("Columns", 0),
        'body_part': data.get('BodyPartExamined', ''),
        'patient_sex': data.get("PatientSex", ''),
        'patient_age': data.get("PatientAge", ''), # data[0x0010,0x1010]
        'modality': data.get('Modality', ''),
        'protocol': data.get('ProtocolName', ''),
        'station_name': data.get('StationName', ''),
        'file_size': os.path.getsize(str(dicom_file)),
    }
        
    return result

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dicom-folder', required=True, type=str)
    parser.add_argument('--output-file', required=True, type=str, 
        help='path to dicom metadata')
    parser.add_argument('--cpus', default=8, type=int, help='Number of cpus to run')
    parser.add_argument('--debug', action='store_true', help='Run on subset of dataset to debug')
    args = parser.parse_args()

    annotations = []
    dicom_files = Path(args.dicom_folder).glob('*m')

    results = []

    if args.debug:
        def early_stop(gen, stop=20):
            for i, e in enumerate(gen):
                if i == stop: break
                yield e
        dicom_files = early_stop(dicom_files, 20)

    executor = Parallel(n_jobs=args.cpus, backend='multiprocessing', prefer='processes', verbose=1)
    do = delayed(process_dicom)
    tasks = (do(dicom_file) for dicom_file in dicom_files)
    results = executor(tasks)

    outfile = args.output_file
    if args.debug:
        outfile = outfile.replace(".csv", ".debug.csv")

    df = pd.DataFrame(results)
    df.to_csv(outfile, index=False)
