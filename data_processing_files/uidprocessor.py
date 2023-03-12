from typing import List

import numpy as np

import pathlib
import json
from upload_utils import PARAMIKO_PROTOCOL

## Class definition of the UIDPROCESSOR class. The Siemens format can be queried using different arguments, among which UID.
# For certain UID values, a certain data structure is returned from MATLAB, this can be the Air Calibration table, the metadata or else the Dose.
# As the data structure returned by MATLAB varies for each UID, one bespoke function has to be designed for each.
# Whilst some functions are redundant, I (Emilien Valat) chose to have a very explicit factory design pattern for better readbility.
# As most data points do not hold data, it is hard to anticipate each data format and hence we raise NotImplementedErrors on data that has yet never been encountered.
# The idea is that each time a new data is "discovered" we implement the right function
# The UIDPROCESSOR object is called by the high level process_UID function

class UIDPROCESSOR:
    def __init__(self, target_save_folder:pathlib.Path, sftp_protocol:PARAMIKO_PROTOCOL) -> None:
        self.target_save_folder= target_save_folder
        ## Set sftp protocol attribute
        self.sftp_protocol = sftp_protocol

    def load_parameter_dict(self):
        if self.UID_key is None:
            raise AttributeError(f'UID_key attribute is None')

        self.parameter_dict_path =self.target_save_folder.joinpath(f'UID_{self.UID_key}.json')

        if self.parameter_dict_path.is_file():
            #print(f'Loading parameter_dict for UID_key {self.UID_key}')
            self.parameter_dict = json.load(open(self.parameter_dict_path, 'r'))
        else:
            self.parameter_dict = {}

    def save_parameter_dict(self):
        with open(self.parameter_dict_path, 'w') as out_file:
            json.dump(self.parameter_dict, out_file)

    def process_member_structure(self, UID_key:int, meta_data, member_id):
        self.accepted_UIDS = [0, 12, 20, 30, 50, 60, 61]
        ## Set UID_key
        if UID_key not in self.accepted_UIDS:
            raise ValueError(f'UID {UID_key} must be in {self.accepted_UIDS}')
        self.UID_key = UID_key
        print(f'Processing {member_id} for UID {self.UID_key}')
        self.load_parameter_dict()
        self.parameter_dict[member_id] = {}
        self.member_id = member_id

        if member_id == 'Version' and self.UID_key==0:
            return self._process_version(meta_data)
        elif member_id == 'Entries' and self.UID_key==0:
            return self._process_entries(meta_data)
        elif member_id == 'ModeParXML' and self.UID_key==0:
            return self._process_mode_par_xml(meta_data)
        elif member_id == 'Lookup' and self.UID_key==0:
            return self._process_lookup(meta_data)
        elif member_id == 'ScanDescr' and self.UID_key==0:
            return self._process_scan_descr(meta_data)
        elif member_id == 'ScanDescrAvg' and self.UID_key==0:
            return self._process_scan_descr_avg(meta_data)
        elif member_id == 'ScanDescrMod' and self.UID_key==0:
            return self._process_scan_descr_mod(meta_data)
        elif member_id == 'ScanDescrOff' and self.UID_key==0:
            return self._process_scan_descr_off(meta_data)
        elif member_id == 'ModeOrg' and self.UID_key==0:
            return self._process_mode_org(meta_data)
        elif member_id == 'Config' and self.UID_key==0:
            return self._process_config(meta_data)

        elif member_id == 'Infoblock' and self.UID_key==12:
            return self._process_info_block(meta_data)
        elif member_id == 'FilePos' and self.UID_key==12:
            return self._process_file_pos(meta_data)
        elif member_id == 'DasHeader' and self.UID_key==12:
            return self._process_das_header(meta_data)
        elif member_id == 'DasFooter' and self.UID_key==12:
            return self._process_das_footer(meta_data)
        elif member_id == 'SliceFooter' and self.UID_key==12:
            return self._process_slice_footer(meta_data)

        elif member_id == 'AcqTimestamp' and self.UID_key==20:
            return self._process_acq_time_stamp(meta_data)
        elif member_id == 'TablePosition' and self.UID_key==20:
            return self._process_table_position(meta_data)
        elif member_id == 'DetA' and self.UID_key==20:
            return self._process_detA(meta_data)
        elif member_id == 'DetB' and self.UID_key==20:
            return self._process_detB(meta_data)
        elif member_id == 'Defect' and self.UID_key==20:
            return self._process_defect(meta_data)
        elif member_id == 'Dose' and self.UID_key==20:
            return self._process_dose(meta_data)
        elif member_id == 'Cardio' and self.UID_key==20:
            return self._process_cardio(meta_data)
        elif member_id == 'Pattern' and self.UID_key==20:
            return self._process_pattern(meta_data)

        elif member_id == 'Type' and self.UID_key==50:
            return self._process_type(meta_data)
        elif member_id == 'PublicHeader' and self.UID_key==50:
            return self._process_public_header(meta_data)
        elif member_id == 'PrivateHeader' and self.UID_key==50:
            return self._process_private_header(meta_data)
        elif member_id == 'Table' and self.UID_key==50:
            return self._process_table(meta_data)
        elif member_id == 'DataI32' and self.UID_key==50:
            return self._process_data_i32(meta_data)
        else:
            raise ValueError(f"Wrong member_id argument")

    ## Process UID 0 Block
    def _process_version(self, meta_data):
        self.parameter_dict[self.member_id] = {}
        struct_names:List[str] = np.dtype(meta_data.dtype).names # type:ignore
        for el, struct_name in zip(meta_data[0,0], struct_names):
            try:
                self.parameter_dict[self.member_id][struct_name] = el.item()
            except ValueError:
                self.parameter_dict[self.member_id][struct_name] = str(el)
        self.save_parameter_dict()

    def _process_entries(self, meta_data):
        struct_names = np.dtype(meta_data.dtype).names
        for row_index, row in enumerate(meta_data[0]):
            self.parameter_dict[self.member_id][f'{row_index}'] = {}
            for element, col_name in zip(row, struct_names): # type:ignore
                self.parameter_dict[self.member_id][f'{row_index}'][col_name] = element[0,0].item()
        self.save_parameter_dict()

    def _process_mode_par_xml(self, meta_data):
        struct_names = np.dtype(meta_data.dtype).names
        for struct, struct_name in zip(meta_data[0,0], struct_names): # type:ignore
            if struct_name == 'Type' or struct_name == 'ModePar':
                self.parameter_dict[self.member_id][struct_name] = {}
                local_struct_names:List[str] = np.dtype(struct[0,0].dtype).names # type:ignore
                for el, el_name in zip(struct[0,0], local_struct_names):
                    try:
                        self.parameter_dict[self.member_id][struct_name][el_name] = el.item()
                    except ValueError:
                        self.parameter_dict[self.member_id][struct_name][el_name] = str(el)
            elif struct_name == 'String':

                local_save_path  = self.target_save_folder.joinpath(f'UID_{self.UID_key}_ModeParXML_String.xml')
                remote_save_path = self.sftp_protocol.remote_dir_path + '/' + f'UID_{self.UID_key}_ModeParXML_String.xml'
                self.sftp_protocol.save_upload(struct[0], local_save_path, remote_save_path)
                self.parameter_dict[self.member_id][struct_name] = remote_save_path
            else:
                raise NotImplementedError (f'Processing of {struct_name} is not implemented')

        self.save_parameter_dict()

    def _process_scan_descr(self, meta_data):
        struct_names = np.dtype(meta_data.dtype).names
        for struct, struct_name in zip(meta_data[0,0], struct_names): # type:ignore

            if struct_name == 'Type':
                self.parameter_dict[self.member_id][struct_name] = {}
                local_struct_names = np.dtype(struct[0,0].dtype).names
                for el, el_name in zip(struct[0,0], local_struct_names): # type:ignore
                    try:
                        self.parameter_dict[self.member_id][struct_name][el_name] = el.item()
                    except ValueError:
                        self.parameter_dict[self.member_id][struct_name][el_name] = str(el)

            elif struct_name =='Det':
                self.parameter_dict[self.member_id][struct_name] = {}
                local_struct_names = np.dtype(struct[0,0].dtype).names
                for element, col_name in zip(struct[0,0], local_struct_names): # type:ignore
                    self.parameter_dict[self.member_id][struct_name][col_name] = element[0].item()

            else:
                try:
                    self.parameter_dict[self.member_id][struct_name] = struct.item()
                except ValueError:
                    # Ugly list to str parsing
                    self.parameter_dict[self.member_id][struct_name] = str(el) # type:ignore

        self.save_parameter_dict()

    def _process_lookup(self, meta_data):
        struct_names = np.dtype(meta_data.dtype).names
        for struct, struct_name in zip(meta_data[0,0], struct_names): # type:ignore
            if struct_name == 'Type':
                self.parameter_dict[self.member_id][struct_name] = {}
                local_struct_names = np.dtype(struct[0,0].dtype).names
                for el, el_name in zip(struct[0,0], local_struct_names): # type:ignore
                    try:
                        self.parameter_dict[self.member_id][struct_name][el_name] = el.item()
                    except ValueError:
                        self.parameter_dict[self.member_id][struct_name][el_name] = str(el)
            else:
                self.parameter_dict[self.member_id][struct_name] = struct[0].tolist()
        self.save_parameter_dict()

    def _process_scan_descr_avg(self, meta_data):
        raise NotImplementedError(f'Function to process {self.member_id} not implemented')

    def _process_scan_descr_mod(self, meta_data):
        raise NotImplementedError(f'Function to process {self.member_id} not implemented')

    def _process_scan_descr_off(self, meta_data):
        raise NotImplementedError(f'Function to process {self.member_id} not implemented')

    def _process_mode_org(self, meta_data):
        raise NotImplementedError(f'Function to process {self.member_id} not implemented')

    def _process_config(self, meta_data):
        raise NotImplementedError(f'Function to process {self.member_id} not implemented')

    ## Process UID 12 Block
    def _process_info_block(self, meta_data):
        struct_names = np.dtype(meta_data.dtype).names
        for struct, struct_name in zip(meta_data[0,0], struct_names): # type:ignore
            if struct_name == 'Status':
                local_struct_names = np.dtype(struct.dtype).names
                self.parameter_dict[self.member_id][struct_name] = {f'{field_name}':field_value[0,0].item() for field_name, field_value in zip(local_struct_names, struct[0,0])} # type:ignore
            elif struct_name == 'NoOf':
                local_struct_names = np.dtype(struct.dtype).names
                self.parameter_dict[self.member_id][struct_name] = {f'{field_name}':field_value[0,0].item() for field_name, field_value in zip(local_struct_names, struct[0,0])} # type:ignore
            elif struct_name == 'A':
                local_struct_names = np.dtype(struct.dtype).names
                self.parameter_dict[self.member_id][struct_name] = {f'{field_name}':field_value[0,0].item() for field_name, field_value in zip(local_struct_names, struct[0,0])} # type:ignore
            elif struct_name == 'B':
                local_struct_names = np.dtype(struct.dtype).names
                self.parameter_dict[self.member_id][struct_name] = {f'{field_name}':field_value[0,0].item() for field_name, field_value in zip(local_struct_names, struct[0,0])} # type:ignore
            else:
                self.parameter_dict[self.member_id][struct_name] = struct[0,0].item()
        self.save_parameter_dict()

    def _process_file_pos(self, meta_data):
        self.parameter_dict[self.member_id] = meta_data[0,0].item()
        self.save_parameter_dict()

    def _process_das_header(self, meta_data):
        struct_names = np.dtype(meta_data.dtype).names
        for el, el_name in zip(meta_data[0,0], struct_names): # type:ignore
            try:
                self.parameter_dict[self.member_id][el_name] = el.item()
            except ValueError:
                self.parameter_dict[self.member_id][el_name] = str(el)
        self.save_parameter_dict()

    def _process_das_footer(self, meta_data):
        struct_names = np.dtype(meta_data.dtype).names
        for el, el_name in zip(meta_data[0,0], struct_names): # type:ignore
            try:
                self.parameter_dict[self.member_id][el_name] = el.item()
            except ValueError:
                self.parameter_dict[self.member_id][el_name] = str(el)
        self.save_parameter_dict()

    def _process_slice_footer(self, meta_data):
        struct_names = np.dtype(meta_data.dtype).names
        for row_index, row in enumerate(meta_data[0]):
            self.parameter_dict[self.member_id][f'{row_index}'] = {}
            for element, col_name in zip(row, struct_names): # type:ignore
                self.parameter_dict[self.member_id][f'{row_index}'][col_name] = element[0,0].item()
        self.save_parameter_dict()

    ## Process UID 20
    def _process_acq_time_stamp(self, meta_data):
        self.parameter_dict[self.member_id] = str(meta_data[0])
        self.save_parameter_dict()

    def _process_table_position(self, meta_data):
        self.parameter_dict[self.member_id] = str(meta_data[0])
        self.save_parameter_dict()

    def _process_detA(self, meta_data):
        self.parameter_dict[self.member_id] = str(meta_data[0])
        self.save_parameter_dict()

    def _process_detB(self, meta_data):
        self.parameter_dict[self.member_id] = str(meta_data[0])
        self.save_parameter_dict()

    def _process_defect(self, meta_data):
        self.parameter_dict[self.member_id] = str(meta_data[0])
        self.save_parameter_dict()

    def _process_dose(self, meta_data):
        self.parameter_dict[self.member_id] = str(meta_data[0])
        self.save_parameter_dict()

    def _process_cardio(self, meta_data):
        self.parameter_dict[self.member_id] = str(meta_data[0])
        self.save_parameter_dict()

    def _process_pattern(self, meta_data):
        self.parameter_dict[self.member_id] = str(meta_data[0])
        self.save_parameter_dict()

    ## Process UID 50
    def _process_type(self, meta_data):
        self.parameter_dict[self.member_id] = meta_data
        self.save_parameter_dict()

    def _process_public_header(self, meta_data):
        self.parameter_dict[self.member_id] = meta_data
        self.save_parameter_dict()

    def _process_private_header(self, meta_data):
        self.parameter_dict[self.member_id] = meta_data
        self.save_parameter_dict()

    def _process_table(self, meta_data):
        local_save_path  = self.target_save_folder.joinpath(f'UID_{self.UID_key}_Table.npy')
        remote_save_path = self.sftp_protocol.remote_dir_path + '/' + f'UID_{self.UID_key}_Table.npy'
        self.sftp_protocol.save_upload(np.array(meta_data), local_save_path, remote_save_path)

    def _process_data_i32(self, meta_data):
        local_save_path  = self.target_save_folder.joinpath(f'UID_{self.UID_key}_DataI32.npy')
        remote_save_path = self.sftp_protocol.remote_dir_path + '/' + f'UID_{self.UID_key}_DataI32.npy'
        self.sftp_protocol.save_upload(np.array(meta_data), local_save_path, remote_save_path)