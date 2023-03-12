"""
This module holds the functions used to upload data from a local machine to the computing resources
"""
import pathlib
from functools import wraps
import xml.etree.ElementTree as ET

import json
import numpy as np
import paramiko

from tqdm import tqdm
## This file is not available to the public
from constants import USER_ID, PASSWORD, MATHS_SERVER

def create_tqdm_callback(*args, **kwargs):
    """Instanciate a tqdm bar and return a callback for paramiko"""
    # Instanciate tqdm bar once
    pbar = tqdm(*args, **kwargs)
    # Create the actual callback
    def view_bar(a, b):
        """Update callback: update total and n (current iteration)"""
        pbar.total = int(b)
        pbar.update(a)
    # Return the callback
    return view_bar

def paramiko_decorator(method):
        @wraps(method)
        def _impl(self, *method_args, **method_kwargs):
            print(f'Opening Paramiko connection to {self.remote_dir_path}')
            self.open_sftp()
            method(self, *method_args, **method_kwargs)
            print(f'Closing Paramiko connection to {self.remote_dir_path}')
            self.client.close()
        return _impl

class PARAMIKO_PROTOCOL():
    def __init__(self, remote_dir_path:str) -> None:
        ## Paramiko business, required to establish the connection
        self.client = paramiko.client.SSHClient() # type: ignore
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.remote_dir_path = remote_dir_path

    def open_sftp(self):
        self.client.connect(MATHS_SERVER, username=USER_ID, password=PASSWORD)
        self.sftp = self.client.open_sftp()

    def close_sftp(self):
        self.client.close()

    def set_remote_dir(self, remote_dir_path:str):
        print(f'Current paramiko protocol dir_path : {self.remote_dir_path}')
        self.remote_dir_path = remote_dir_path
        print(f'Updated paramiko protocol dir_path : {self.remote_dir_path}')

    @paramiko_decorator
    def sftp_transfer(self, local_file_path:pathlib.Path, remote_file_path:str):
        print(f'Copying file from {local_file_path} to {remote_file_path}')
        view_bar = create_tqdm_callback(ascii=True,unit='b',unit_scale=True)
        self.sftp.put(str(local_file_path), remote_file_path, callback=view_bar)

    @paramiko_decorator
    def mkdir(self, dir_path:str):
        try:
            self.sftp.stat(dir_path)
            print('Folder already exists, passing...')
        except FileNotFoundError:
            print(f'Making remote dir at {dir_path}')
            self.sftp.mkdir(dir_path)

    # Does not work!
    @paramiko_decorator
    def write_data(self, data, file_name:str, file_extension:str):
        target_file_path = self.remote_dir_path +'/' +file_name
        with self.sftp.open(target_file_path, "w") as remote_file_path:
            print(remote_file_path)
            print(f'Uploading {file_name} at {target_file_path}')
            if file_extension == '.npy':
                np.save(remote_file_path, data)
            elif file_extension == '.json':
                json.dump(data, remote_file_path)
            else:
                raise NotImplementedError (f'Not implemented for file extension {file_extension}')

    def save_upload(self, data, local_save_path:pathlib.Path, remote_save_path:str):
        if local_save_path.suffix == '.npy':
            np.save(local_save_path, data)

        elif local_save_path.suffix == '.json':
            json.dump(data, open(local_save_path))

        elif local_save_path.suffix == '.xml':
            tree = ET.ElementTree(ET.fromstring(data))
            tree.write(local_save_path)

        else:
            raise NotImplementedError (f'Not implemented for file extension {local_save_path.suffix}')

        self.sftp_transfer(local_save_path, remote_save_path)


