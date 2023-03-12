import numpy as np
import pathlib
import json

from uidprocessor import UIDPROCESSOR
from matlabbridge import MATLABBRIDGE
from upload_utils import PARAMIKO_PROTOCOL

## Class definition of the PTRFILEPROCESSOR that processes a .ptr file
## It iterates over the arguments of the matlab scripts that extract the data and parses the latter into a "friendly format"
class PTRFILEPROCESSOR:
    def __init__(self, ptr_folder_path:pathlib.Path) -> None:
        print(f'Instanciate PTRFILEPROCESSOR object for folder {ptr_folder_path}')
        self.ptr_folder_path = pathlib.Path(ptr_folder_path)
        self.matlab_bridge = MATLABBRIDGE(self.ptr_folder_path)

    def process_header_file(self, sftp_protocol:PARAMIKO_PROTOCOL):
        local_save_path = self.matlab_bridge.save_folder_path.joinpath('headers.json')
        self.matlab_bridge.fetch_header(local_save_path)
        remote_save_path = sftp_protocol.remote_dir_path + '/headers.jon'
        sftp_protocol.sftp_transfer(local_save_path, remote_save_path)

    def process_wrapper_bytes(self, sftp_protocol:PARAMIKO_PROTOCOL):
        data = self.matlab_bridge.fetch_wrapper_bytes()
        local_save_path  = self.matlab_bridge.save_folder_path.joinpath('wrapper_bytes.npy')
        remote_save_path = sftp_protocol.remote_dir_path + '/wrapper_bytes.npy'
        sftp_protocol.save_upload(data, local_save_path, remote_save_path)

    def process_det_no(self, det_no_key:int, sftp_protocol:PARAMIKO_PROTOCOL):
        data = self.matlab_bridge.fetch_det_no( det_no_key)
        local_save_path  = self.matlab_bridge.save_folder_path.joinpath(f'det_no_{det_no_key}.npy')
        remote_save_path = sftp_protocol.remote_dir_path + f'/det_no_{det_no_key}.npy'
        sftp_protocol.save_upload(data, local_save_path, remote_save_path)

    def process_table_no(self, table_no_key:int, sftp_protocol:PARAMIKO_PROTOCOL):
        data = self.matlab_bridge.fetch_det_no( table_no_key)
        local_save_path  = self.matlab_bridge.save_folder_path.joinpath(f'table_no_{table_no_key}.npy')
        remote_save_path = sftp_protocol.remote_dir_path + f'/table_no_{table_no_key}.npy'
        sftp_protocol.save_upload(data, local_save_path, remote_save_path)

    def get_n_detector_row(self, file_name:pathlib.Path):
        meta_data = json.load(open(f'{file_name.with_suffix("")}/UID_0.json'))
        n_det_row = meta_data['ScanDescr']['Type']['Slices']+1-meta_data['ScanDescr']['Type']['SlicesMon']
        print(f"N Detector Row = {n_det_row}")
        return n_det_row

    def get_n_readings(self):
        return self.matlab_bridge.fetch_n_readings()

    def process_UID(self, UID_key:int, sftp_protocol:PARAMIKO_PROTOCOL):

        UID_data = self.matlab_bridge.fetch_UID(UID_key)

        if len(UID_data)==0:
            print(f'No entry for UID {UID_key}')

        else:
            if UID_key == 0:
                member_structures = np.dtype(UID_data.dtype).names
                for index, member_structure in enumerate(member_structures): # type:ignore
                    self.uid_processor.process_member_structure(UID_key, UID_data[0,0][index], member_structure)

            elif UID_key == 12:
                data, header = UID_data[0], UID_data[1]
                ## First, we save the data as a np file
                sftp_protocol.write_data(data, 'UID_12_data.npy', '.npy')
                local_save_path  = self.matlab_bridge.save_folder_path.joinpath('UID_12_data.npy')
                remote_save_path = sftp_protocol.remote_dir_path + '/UID_12_data.npy'
                sftp_protocol.save_upload(data, local_save_path, remote_save_path)
                ## Then, we process the header as we would for a matlab struct
                member_structures = np.dtype(header[0,0].dtype).names
                for index, member_structure in enumerate(member_structures): # type:ignore
                    # Don't ask me what this horror is header[0][0][0][0][index], it's nested format returned from the .mat file
                    self.uid_processor.process_member_structure(UID_key, header[0][0][0][0][index], member_structure)

            elif UID_key == 20:
                for key, value in UID_data.items():# type:ignore
                    self.uid_processor.process_member_structure(UID_key, value, key)

            elif UID_key == 50:
                for cell in UID_data:
                    try:
                        member_structures = cell.keys()
                        for member_structure in member_structures:
                            self.uid_processor.process_member_structure(UID_key, cell[member_structure], member_structure)
                    except AttributeError:
                        print('Empty cell, passing')

            sftp_protocol.sftp_transfer(self.uid_processor.parameter_dict_path, sftp_protocol.remote_dir_path + '/'+ self.uid_processor.parameter_dict_path.name)

    def determine_body_part(self, ptr_folder_path:pathlib.Path) -> dict:
        ptr_files_list = list(ptr_folder_path.glob('*.ptr'))
        ptr_files_dict = {file_name : {'n_readings':self.get_n_readings(), 'body_part':'chest'} for file_name in ptr_files_list}
        if ptr_files_list[0] < ptr_files_list[1]:
            ptr_files_dict[ptr_files_list[1]]['body_part'] = 'abdomen'
        else:
            ptr_files_dict[ptr_files_list[0]]['body_part'] = 'abdomen'
        assert(ptr_files_dict[ptr_files_list[0]]['body_part'] != ptr_files_dict[ptr_files_list[1]]['body_part'])
        return ptr_files_dict

    def process_ptr_file(self, file_name:pathlib.Path, body_part_name:str, sftp_protocol:PARAMIKO_PROTOCOL):
        # Necessary to set so the MATLABBRIDGE knows where the file is
        self.matlab_bridge.set_current_ptr_file_path(file_name)
        # Necessary to set so the MATLABBRIDGE object knows where to save the meta data
        current_save_folder_path = self.ptr_folder_path.joinpath(body_part_name)
        self.matlab_bridge.set_save_folder_path(current_save_folder_path)

        self.uid_processor = UIDPROCESSOR(current_save_folder_path, sftp_protocol)

        self.process_wrapper_bytes(sftp_protocol)
        self.process_det_no(1, sftp_protocol)
        self.process_det_no(2, sftp_protocol)
        self.process_table_no(1, sftp_protocol)
        self.process_table_no(2, sftp_protocol)

        for UID_key in [0, 12, 20, 30, 50, 60, 61]:
            self.process_UID(UID_key, sftp_protocol)

        self.process_header_file(sftp_protocol)

        self.matlab_bridge.fetch_readings(self.get_n_readings(), self.get_n_detector_row(file_name), sftp_protocol)
