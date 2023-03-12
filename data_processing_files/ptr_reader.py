"""
This module holds the functions used to read data from the proprietary .ptr format.
Note that the core matlab functions are not provided, hence this module cannot be used as is.
"""
import pathlib

from upload_utils import PARAMIKO_PROTOCOL
from ptrfileprocessor import PTRFILEPROCESSOR

def process_raw_folder(ptr_folder_path:pathlib.Path, sftp_protocol:PARAMIKO_PROTOCOL):
    print(f'Processing folder {ptr_folder_path}')
    assert ptr_folder_path.is_dir()
    ## Create File Processor object
    files_processor = PTRFILEPROCESSOR(ptr_folder_path)
    file_name_to_bodypart_dict = files_processor.determine_body_part(ptr_folder_path)

    remote_folder_path = sftp_protocol.remote_dir_path

    for ptr_file_name, bodypart_dict in file_name_to_bodypart_dict.items():
        print('+'+100*'-'+'+')
        print(f"Processing file {ptr_file_name}")
        current_body_part = bodypart_dict['body_part']
        print(f"It is the {current_body_part}, with {bodypart_dict['n_readings']} readings.")
        remote_body_part_path = f'{remote_folder_path}/{current_body_part}'
        sftp_protocol.mkdir(remote_body_part_path)
        sftp_protocol.remote_dir_path = remote_body_part_path
        local_dir_path = ptr_folder_path.joinpath(current_body_part)
        print(f"Making local dir at {local_dir_path}")
        local_dir_path.mkdir(exist_ok=True)
        files_processor.process_ptr_file(ptr_file_name, current_body_part, sftp_protocol)

if __name__ == '__main__':
    functions_folder_path = r'C:\Users\hx21262\all_in_one_v4\data_processing_files'
    body_part = 'chest'
    remote_folder_path = f'/store/DAMTP/ev373/all_in_one/XNAT_S03362/XNAT_E04148/1135_20200224/{body_part}'
    paramiko_protocol = PARAMIKO_PROTOCOL(remote_folder_path)

    ptr_file_name = pathlib.Path(r'C:\Users\hx21262\Downloads\temp_dir_xnat\1135_20200224\resources\RAW\files\66862.Anonymous.CT..601.RAW.20200224.155058.683027.2020.03.10.15.12.24.675000.1249019438.ptr')

    ptr_folder_path = pathlib.Path(r'C:\Users\hx21262\Downloads\temp_dir_xnat\1135_20200224\resources\RAW\files')
    #process_raw_folder(ptr_folder_path, paramiko_protocol)
    files_processor = PTRFILEPROCESSOR(ptr_folder_path)
    files_processor.process_ptr_file(ptr_file_name, body_part, paramiko_protocol)



