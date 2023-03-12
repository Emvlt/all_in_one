"""
This module holds the functions used to iterate over the available data in the XNAT server and process one experiment at once.
"""
import pathlib
from typing import Dict
from datetime import datetime

import xnat
import json

from constants import PROJECT_PATH, XNAT_HOST ,PROJECT_ID, USER_ID, PASSWORD, MATHS_DATASET_PATH
from download_utils import download_raw_data, unzip_file
from ptr_reader import process_raw_folder
from upload_utils import PARAMIKO_PROTOCOL

## This is the pipeline to process ONE (1) experiment.
# 1) Downloads the data
# 2) Unzips the file and deletes the .zip file
# 3) Processes the file (extracts the content of the .ptr file to friendly format)
def process_scan(project, subject:str, experiment:str, experiment_label:str, download_folder_path:pathlib.Path, paramiko_protocol:PARAMIKO_PROTOCOL, debug = False):
    ## Download data
    zip_file_path = download_raw_data(project, subject, experiment, experiment_label, download_folder_path, debug)
    ## Unzip file
    temp_dir_path = unzip_file(zip_file_path, zip_file_path.parent, debug)
    ## Process file
    process_raw_folder(temp_dir_path, paramiko_protocol)

## This is the pipeline to process ALL the experiments for all the subjects of the project;
# simple loop over all the available data
def process_all_scans(connection, project_id:str, download_folder_path:pathlib.Path, progress_dict:Dict, progress_dict_path:pathlib.Path, debug=False):
    project     = connection.projects[project_id]
    subjects = project.subjects
    ## Instanciate Paramiko connection
    paramiko_protocol = PARAMIKO_PROTOCOL(MATHS_DATASET_PATH)
    for subject in subjects:
        print(f'Subject: {subject}')
        progress_dict[subject] = {}
        experiments = project.subjects[subject].experiments
        subject_path_str = f'{MATHS_DATASET_PATH}/{subject}'
        paramiko_protocol.mkdir(subject_path_str)
        for experiment in experiments:
            print(f'Experiment: {experiment}')
            if experiment in progress_dict[subject].keys():
                print(f'Experiment {experiment} for subject {subject} already uploaded, passing...')
            else:
                experiment_path_str = f'{subject_path_str}/{experiment}'
                paramiko_protocol.mkdir(experiment_path_str)
                experiment_label = project.experiments[experiment].label
                ## Make remote dir
                remote_dir_path = f'{experiment_path_str}/{experiment_label}'
                paramiko_protocol.mkdir(remote_dir_path)
                paramiko_protocol.set_remote_dir(remote_dir_path)
                process_scan(project, subject, experiment, experiment_label, download_folder_path, paramiko_protocol, debug)
                progress_dict[subject][experiment] = f'uploaded on {datetime.now()}'
                with open(progress_dict_path, 'w') as out_file:
                    json.dump(progress_dict, out_file)

if __name__=='__main__':
    debug = False
    progress_dict_path = pathlib.Path('upload_progress_dict.json')
    if progress_dict_path.is_file():
        progress_dict = json.load(open(progress_dict_path))
    else:
        progress_dict = {}
    progress_dict = {}
    with xnat.connect(XNAT_HOST, user=USER_ID, password=PASSWORD) as connection:
        process_all_scans(connection, PROJECT_ID, PROJECT_PATH, progress_dict, progress_dict_path, debug=debug)
