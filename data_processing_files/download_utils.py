"""
This module holds the functions used to download data from the XNAT server
"""
import pathlib

from tqdm import tqdm

## function that downloads the RAW resources for the given 'experiment', for the given 'subject', for the given 'project'.
# The XNAT folder is organised as follows:
# project
# ---- subjects
#      ---- experiments
#           ---- resources
#                ---- RAW
def download_raw_data(project, subject:str, experiment:str, experiment_label:str, download_folder_path:pathlib.Path, debug = False):
    ## First, get the resource object from the project XNAT object.
    resource = project.subjects[subject].experiments[experiment].resources['RAW']
    ## Second, create the download file path locally
    zip_file_path = download_folder_path.joinpath(f'{experiment_label}.zip')
    ## Download raw file
    # assert if the file already exists locally
    if zip_file_path.is_file():
        print(f'File {experiment_label} already exists in path {download_folder_path}, processing...')
    else:
        # if not, download it
        print(f'Downloading raw {experiment_label} file from XNAT server...')
        if not debug:
            tqdm(resource.download(zip_file_path))
        print('download ended')
    # Return the address of the file
    return zip_file_path

