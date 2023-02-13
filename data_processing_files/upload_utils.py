"""
This module holds the functions used to upload data from a local machine to the computing resources
"""
import pathlib
from functools import wraps

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
            print(f'Opening Paramiko connection to {MATHS_SERVER}')
            self.client.connect(MATHS_SERVER, username=USER_ID, password=PASSWORD)
            self.sftp = self.client.open_sftp()
            method(self, *method_args, **method_kwargs)
            print(f'Closing Paramiko connection to {MATHS_SERVER}')
            self.client.close()
        return _impl

class PARAMIKO_PROTOCOL():
    def __init__(self, debug:bool) -> None:
        ## Paramiko business, required to establish the connection
        self.client = paramiko.client.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.set_debug_mode(debug)

    def set_debug_mode(self, debug):
        self.debug = debug

    @paramiko_decorator
    def sftp_transfer(self, local_file_path:pathlib.Path, remote_file_path:pathlib.Path):
        print(f'Copying file from {local_file_path} to {remote_file_path}')
        if not self.debug:
            view_bar = create_tqdm_callback(ascii=True,unit='b',unit_scale=True)
            self.sftp.put(str(local_file_path), str(remote_file_path), callback=view_bar)

    @paramiko_decorator
    def mkdir(self, dir_path:pathlib.Path):
        print(f'Making dir at {dir_path}')
        if not self.debug:
            self.sftp.mkdir(str(dir_path))
