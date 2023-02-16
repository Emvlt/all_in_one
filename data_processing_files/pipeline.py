"""
This module holds the functions used to iterate over the available data in the XNAT server and process one experiment at once.
"""

from typing import Dict
import pathlib
from datetime import datetime
import zipfile
import shutil

import json
import xnat

from constants import PROJECT_PATH, XNAT_HOST ,PROJECT_ID, USER_ID, PASSWORD, MATHS_DATASET_PATH
from download_utils import download_raw_data
from ptr_reader import process_raw_file
from upload_utils import PARAMIKO_PROTOCOL

## Unzips and deletes .zip file to keep things tidy; returns the address of the .ptr files folder
def unzip_file(file_path:pathlib.Path, unzip_dir_path:pathlib.Path, debug=False) -> pathlib.Path:
    assert file_path.suffix == '.zip'
    print(f'Unpacking zip file to {unzip_dir_path}')
    if not debug:
        with zipfile.ZipFile(str(file_path), 'r') as zip_ref:
            zip_ref.extractall(unzip_dir_path)
        file_path.unlink()
    return file_path.with_suffix('').joinpath('resources/RAW/files')

def zip_file(temp_dir_path:pathlib.Path, debug=False):
    print(f'Zipping {temp_dir_path}')
    if not debug:
        shutil.make_archive(temp_dir_path, 'zip', temp_dir_path)
    return temp_dir_path.with_suffix('.zip')

## Returns a status_file, a dictionary containing the status of each experiment (downloaded/processed/reuploaded)
## In case of interruption of download
def open_status_file() -> Dict:
    if PROJECT_PATH.joinpath('status.json').is_file():
        return json.load(open(PROJECT_PATH.joinpath('status.json')))
    else:
        return {}

def save_status_file(json_dict:Dict):
    with open(PROJECT_PATH.joinpath('status.json'), 'w') as file_path:
        json.dump(json_dict, file_path, indent=4)

## This is the pipeline to process ONE (1) experiment.
# 1) Downloads the data
# 2) Unzips the file and deletes the .zip file
# 3) Processes the file (extracts the content of the .ptr file to friendly format)
# 4) Zips the extracted data before sending to computing resources
# 5) Uploads the new .zip file to the computing resources
def process_scan(project, subject:str, experiment:str, experiment_label:str, download_folder_path:pathlib.Path, paramiko_protocol:PARAMIKO_PROTOCOL, debug = False, packed_readings = True):
    status_dict = {}
    ## Stamp Access
    now = datetime.now()
    status_dict['accessed'] = now.strftime("%m/%d/%Y/%H:%M:%S")
    ## Make remote dir
    remote_dir_path = f'{MATHS_DATASET_PATH}/{subject}/{experiment}/{experiment_label}'
    paramiko_protocol.mkdir(remote_dir_path)
    paramiko_protocol.set_remote_dir(remote_dir_path)
    ## Download data
    zip_file_path = download_raw_data(project, subject, experiment, experiment_label, download_folder_path, debug)
    ## Stamp Downloaded
    now = datetime.now()
    status_dict['downloaded'] = now.strftime("%m/%d/%Y/%H:%M:%S")
    ## Unzip file
    temp_dir_path = unzip_file(zip_file_path, zip_file_path.parent, debug)
    ## Process file
    process_raw_file(temp_dir_path, debug, packed_readings, sftp_protocol = paramiko_protocol)
    ## Stamp Processed
    now = datetime.now()
    status_dict['processed'] = now.strftime("%m/%d/%Y/%H:%M:%S")
    ## Zip dir
    zipped_file_path = zip_file(temp_dir_path, debug)
    ## Upload dir
    paramiko_protocol.sftp_transfer(zipped_file_path, f'{MATHS_DATASET_PATH}/{subject}/{experiment}/{zip_file_path.stem}.zip')
    ## Stamp Uploaded
    now = datetime.now()
    status_dict['uploaded'] = now.strftime("%m/%d/%Y/%H:%M:%S")
    return status_dict

## This is the pipeline to process ALL the experiments for all the subjects of the project;
# simple loop over all the available data
def process_all_scans(connection, project_id:str, download_folder_path:pathlib.Path, debug=False):
    status_dict = open_status_file()
    project     = connection.projects[project_id]
    subjects = project.subjects
    ## Instanciate Paramiko connection
    paramiko_protocol = PARAMIKO_PROTOCOL(debug)
    for subject in subjects:
        print(f'Subject: {subject}')
        experiments = project.subjects[subject].experiments
        status_dict[subject] = {}
        paramiko_protocol.mkdir(f'{MATHS_DATASET_PATH}/{subject}')
        for experiment in experiments:
            print(f'Experiment: {experiment}')
            paramiko_protocol.mkdir(f'{MATHS_DATASET_PATH}/{subject}/{experiment}')
            experiment_label = project.experiments[experiment].label
            returned_status_dict = process_scan(project, subject, experiment, experiment_label, download_folder_path, paramiko_protocol, debug)
            status_dict[subject][experiment_label] = returned_status_dict
            save_status_file(status_dict)

if __name__=='__main__':
    debug = True
    with xnat.connect(XNAT_HOST, user=USER_ID, password=PASSWORD) as connection:
        process_all_scans(connection, PROJECT_ID, PROJECT_PATH, debug=debug)
