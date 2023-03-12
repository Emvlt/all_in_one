"""
This module holds the functions used to download data from the XNAT server
"""
import pathlib
import zipfile

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

## Unzips and deletes .zip file to keep things tidy; returns the address of the .ptr files folder
def unzip_file(file_path:pathlib.Path, unzip_dir_path:pathlib.Path, debug=False) -> pathlib.Path:
    assert file_path.suffix == '.zip'
    unzipped_folder_path = file_path.with_suffix('').joinpath('resources/RAW/files')
    if unzipped_folder_path.is_file():
        print(f'Folder already unzipped...')
    else:
        if not debug:
            print(f'Unpacking zip file to {unzip_dir_path}')
            with zipfile.ZipFile(str(file_path), 'r') as zip_ref:
                zip_ref.extractall(unzip_dir_path)
            file_path.unlink()
    print(f"Returning unzipped folders at {unzipped_folder_path}")
    return unzipped_folder_path