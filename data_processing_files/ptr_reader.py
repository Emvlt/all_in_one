"""
This module holds the functions used to read data from the proprietary .ptr format.
Note that the core matlab functions are not provided, hence this module cannot be used as is.
"""

import scipy
import matlab.engine
import numpy as np
from tqdm import tqdm
import pathlib
import json
import xml.etree.ElementTree as ET
from constants import MATLAB_FUNCS_DIR

## Class definition of the UIDPROCESSOR class. The Siemens format can be queried using different arguments, among which UID.
# For certain UID values, a certain data structure is returned from MATLAB, this can be the Air Calibration table, the metadata or else the Dose.
# As the data structure returned by MATLAB varies for each UID, one bespoke function has to be designed for each.
# Whilst some functions are redundant, I (Emilien Valat) chose to have a very explicit factory design pattern for better readbility.
# As most data points do not hold data, it is hard to anticipate each data format and hence we raise NotImplementedErrors on data that has yet never been encountered.
# The idea is that each time a new data is "discovered" we implement the right function
# The UIDPROCESSOR object is called by the high level process_UID function
class UIDPROCESSOR:
    def __init__(self, file_name:pathlib.Path, UID_key:int) -> None:
        self.file_name:pathlib.Path = file_name
        self.accepted_UIDS = [0, 12, 20, 30, 50, 60, 61]
        ## Set UID_key
        if UID_key not in self.accepted_UIDS:
            raise ValueError(f'UID {UID_key} must be in {self.accepted_UIDS}')
        self.UID_key = UID_key
        ## Load parameter dict
        self.load_parameter_dict()

    def load_parameter_dict(self):
        if self.UID_key is None:
            raise AttributeError(f'UID_key attribute is None')
        self.parameter_dict_path = pathlib.Path(f'{self.file_name.with_suffix("")}/UID_{self.UID_key}.json')
        if self.parameter_dict_path.is_file():
            #print(f'Loading parameter_dict for UID_key {self.UID_key}')
            self.parameter_dict = json.load(open(self.parameter_dict_path, 'r'))
        else:
            self.parameter_dict = {}

    def save_parameter_dict(self):
        with open(self.parameter_dict_path, 'w') as out_file:
            json.dump(self.parameter_dict, out_file)

    def process_member_structure(self, meta_data, member_id):
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
        struct_names = np.dtype(meta_data.dtype).names
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
            for element, col_name in zip(row, struct_names):
                self.parameter_dict[self.member_id][f'{row_index}'][col_name] = element[0,0].item()
        self.save_parameter_dict()

    def _process_mode_par_xml(self, meta_data):
        struct_names = np.dtype(meta_data.dtype).names
        for struct, struct_name in zip(meta_data[0,0], struct_names):
            if struct_name == 'Type' or struct_name == 'ModePar':
                self.parameter_dict[self.member_id][struct_name] = {}
                local_struct_names = np.dtype(struct[0,0].dtype).names
                for el, el_name in zip(struct[0,0], local_struct_names):
                    try:
                        self.parameter_dict[self.member_id][struct_name][el_name] = el.item()
                    except ValueError:
                        self.parameter_dict[self.member_id][struct_name][el_name] = str(el)
            elif struct_name == 'String':
                tree = ET.ElementTree(ET.fromstring(struct[0]))
                tree.write(str(pathlib.Path(self.file_name.with_suffix("").joinpath(f'UID_{self.UID_key}_ModeParXML_String.xml'))))
                self.parameter_dict[self.member_id][struct_name] = f'{self.file_name.with_suffix("")}/UID_{self.UID_key}_ModeParXML_String.xml'
            else:
                raise NotImplementedError (f'Processing of {struct_name} is not implemented')

        self.save_parameter_dict()

    def _process_scan_descr(self, meta_data):
        struct_names = np.dtype(meta_data.dtype).names
        for struct, struct_name in zip(meta_data[0,0], struct_names):

            if struct_name == 'Type':
                self.parameter_dict[self.member_id][struct_name] = {}
                local_struct_names = np.dtype(struct[0,0].dtype).names
                for el, el_name in zip(struct[0,0], local_struct_names):
                    try:
                        self.parameter_dict[self.member_id][struct_name][el_name] = el.item()
                    except ValueError:
                        self.parameter_dict[self.member_id][struct_name][el_name] = str(el)

            elif struct_name =='Det':
                self.parameter_dict[self.member_id][struct_name] = {}
                local_struct_names = np.dtype(struct[0,0].dtype).names
                for element, col_name in zip(struct[0,0], local_struct_names):
                    self.parameter_dict[self.member_id][struct_name][col_name] = element[0].item()

            else:
                try:
                    self.parameter_dict[self.member_id][struct_name] = struct.item()
                except ValueError:
                    self.parameter_dict[self.member_id][struct_name] = str(el)

        self.save_parameter_dict()

    def _process_lookup(self, meta_data):
        struct_names = np.dtype(meta_data.dtype).names
        for struct, struct_name in zip(meta_data[0,0], struct_names):
            if struct_name == 'Type':
                self.parameter_dict[self.member_id][struct_name] = {}
                local_struct_names = np.dtype(struct[0,0].dtype).names
                for el, el_name in zip(struct[0,0], local_struct_names):
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
        for struct, struct_name in zip(meta_data[0,0], struct_names):
            if struct_name == 'Status':
                local_struct_names = np.dtype(struct.dtype).names
                self.parameter_dict[self.member_id][struct_name] = {f'{field_name}':field_value[0,0].item() for field_name, field_value in zip(local_struct_names, struct[0,0])}
            elif struct_name == 'NoOf':
                local_struct_names = np.dtype(struct.dtype).names
                self.parameter_dict[self.member_id][struct_name] = {f'{field_name}':field_value[0,0].item() for field_name, field_value in zip(local_struct_names, struct[0,0])}
            elif struct_name == 'A':
                local_struct_names = np.dtype(struct.dtype).names
                self.parameter_dict[self.member_id][struct_name] = {f'{field_name}':field_value[0,0].item() for field_name, field_value in zip(local_struct_names, struct[0,0])}
            elif struct_name == 'B':
                local_struct_names = np.dtype(struct.dtype).names
                self.parameter_dict[self.member_id][struct_name] = {f'{field_name}':field_value[0,0].item() for field_name, field_value in zip(local_struct_names, struct[0,0])}
            else:
                self.parameter_dict[self.member_id][struct_name] = struct[0,0].item()
        self.save_parameter_dict()

    def _process_file_pos(self, meta_data):
        self.parameter_dict[self.member_id] = meta_data[0,0].item()
        self.save_parameter_dict()

    def _process_das_header(self, meta_data):
        struct_names = np.dtype(meta_data.dtype).names
        for el, el_name in zip(meta_data[0,0], struct_names):
            try:
                self.parameter_dict[self.member_id][el_name] = el.item()
            except ValueError:
                self.parameter_dict[self.member_id][el_name] = str(el)
        self.save_parameter_dict()

    def _process_das_footer(self, meta_data):
        struct_names = np.dtype(meta_data.dtype).names
        for el, el_name in zip(meta_data[0,0], struct_names):
            try:
                self.parameter_dict[self.member_id][el_name] = el.item()
            except ValueError:
                self.parameter_dict[self.member_id][el_name] = str(el)
        self.save_parameter_dict()

    def _process_slice_footer(self, meta_data):
        struct_names = np.dtype(meta_data.dtype).names
        for row_index, row in enumerate(meta_data[0]):
            self.parameter_dict[self.member_id][f'{row_index}'] = {}
            for element, col_name in zip(row, struct_names):
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
        np.save(f'{self.file_name.with_suffix("")}/UID_{self.UID_key}_Table.npy', np.array(meta_data))

    def _process_data_i32(self, meta_data):
        np.save(f'{self.file_name.with_suffix("")}/UID_{self.UID_key}_DataI32.npy', np.array(meta_data))

## The MATLABBRIDGE class is an interface between the Python script and the Matlab engine that is *required* to read the proprietary Siemens data format
# It requires the path of the folder in which there are the files we want to extract, and (implicitly) the folder where the said matlab functions are stored
# Whilst some functions are redundant, I (Emilien Valat) chose to have a very explicit factory design pattern for better readbility.
class MATLABBRIDGE:
    def __init__(self, folder_path:pathlib.Path) -> None:
        self.folder_path = folder_path
        self.eng = None
        self.eng = matlab.engine.start_matlab()
        self.eng.cd(str(self.folder_path), nargout=0)
        self.eng.addpath(self.eng.genpath(str(MATLAB_FUNCS_DIR)), nargout=0)

    def set_save_folder_path(self, save_folder_path:pathlib.Path):
        self.save_folder_path = save_folder_path

    def fetch_image(self, file_name:pathlib.Path, image_no:int) -> np.array:
        return self.eng.read_xacb2(str(self.folder_path.joinpath(file_name)), 'ImageNo', float(image_no))[0]

    def fetch_scan(self, file_name:pathlib.Path, scan_no:int) -> np.array:
        return self.eng.read_xacb2(str(self.folder_path.joinpath(file_name)), 'ScanNo', float(scan_no))[0]

    def fetch_reading(self, file_name:pathlib.Path, scan_no:int, reading_no:int) -> np.array:
        return self.eng.read_xacb2(str(self.folder_path.joinpath(file_name)), 'ReadingNo', float(reading_no), 'ScanNo', float(scan_no))[0]

    def fetch_scans(self, file_name:pathlib.Path, n_readings:int):
        raise NotImplementedError

    def fetch_images(self, file_name:pathlib.Path, n_readings:int):
        raise NotImplementedError

    def fetch_readings(self, file_name:pathlib.Path, n_readings:int, n_detector_rows:int):
        complete_reading = np.zeros((736, n_readings*n_detector_rows))
        for scan_no in range(1,2):
            for reading_no in tqdm(range(1, 1+n_readings)):
                complete_reading[:,(reading_no-1)*n_detector_rows:(reading_no)*n_detector_rows] = self.fetch_reading(file_name, scan_no, reading_no)
        np.save(self.save_folder_path.joinpath(f'readings'), complete_reading,allow_pickle=True)

    def mat_file_temp_save(self, file_name:pathlib.Path, uid_key:int):
        metadata_data_file_name = self.save_folder_path.joinpath(f'UID_{uid_key}.mat')
        if not pathlib.Path(metadata_data_file_name).is_file():
            print(f'No file {metadata_data_file_name}, fetching from .ptr file and saving to .mat file ...')
            self.eng.read_UID(str(self.save_folder_path.joinpath(file_name)), uid_key, nargout=0)
        print(f'Reading meta data file at {metadata_data_file_name}')
        return scipy.io.loadmat(metadata_data_file_name)['UID_data']

    def fetch_UID_0(self, file_name:pathlib.Path) -> np.array:
        ## Here, we have to save a temp .mat file as we get the error: "ValueError: only a scalar struct can be returned from MATLAB"
        return self.mat_file_temp_save(file_name, 0)

    def fetch_UID_12(self, file_name:pathlib.Path) -> np.array:
        ## Here, we have to save a temp .mat file as we get the error: "ValueError: only a scalar struct can be returned from MATLAB"
        ## Having the default nargout = 1 will return the data part of UID 12 only
        data   = self.eng.return_UID(str(self.folder_path.joinpath(file_name)), 12, nargout = 1)
        header = self.mat_file_temp_save(file_name, 12)
        return data, header

    def fetch_UID_20(self, file_name:pathlib.Path) -> np.array:
        return self.eng.return_UID(str(self.folder_path.joinpath(file_name)), 20, nargout = 1)

    def fetch_UID_30(self, file_name:pathlib.Path) -> np.array:
        return self.eng.return_UID(str(self.folder_path.joinpath(file_name)), 30, nargout = 1)

    def fetch_UID_50(self, file_name:pathlib.Path) -> np.array:
        return self.eng.return_UID(str(self.folder_path.joinpath(file_name)), 50, nargout = 1)

    def fetch_UID_60(self, file_name:pathlib.Path) -> np.array:
        return self.eng.return_UID(str(self.folder_path.joinpath(file_name)), 60, nargout = 1)

    def fetch_UID_61(self, file_name:pathlib.Path) -> np.array:
        return self.eng.return_UID(str(self.folder_path.joinpath(file_name)), 61, nargout = 1)

    def fetch_UID(self, file_name:pathlib.Path, UID_key:int) -> np.array:
        if UID_key == 0:
            return self.fetch_UID_0(file_name)
        elif UID_key == 12:
            return self.fetch_UID_12(file_name)
        elif UID_key == 20:
            return self.fetch_UID_20(file_name)
        elif UID_key == 30:
            return self.fetch_UID_30(file_name)
        elif UID_key == 50:
            return self.fetch_UID_50(file_name)
        elif UID_key == 60:
            return self.fetch_UID_60(file_name)
        elif UID_key == 61:
            return self.fetch_UID_61(file_name)
        else:
            raise NotImplementedError(f'Fetch function not implemented for {UID_key}')

    def fetch_wrapper_bytes(self, file_name:pathlib.Path):
        return self.eng.read_xacb2(str(self.folder_path.joinpath(file_name)), 'wrapperBytes', 0.0)[0]

    def fetch_det_no(self, file_name:pathlib.Path, det_no_key:int):
        return self.eng.read_xacb2(str(self.folder_path.joinpath(file_name)), 'DetNo', float(det_no_key))[0]

    def fetch_table_no(self, file_name:pathlib.Path, table_no_key:int):
        return self.eng.read_xacb2(str(self.folder_path.joinpath(file_name)), 'TableNo', float(table_no_key))[0]

## Class definition of the PTRFILEPROCESSOR that processes a .ptr file
## It iterates over the arguments of the matlab scripts that extract the data and parses the latter into a "friendly format"
class PTRFILEPROCESSOR:
    def __init__(self, folder_path:str) -> None:
        self.folder_path = pathlib.Path(folder_path)
        self.matlab_bridge = MATLABBRIDGE(self.folder_path)

    def file_extension_check(self, full_path:pathlib.Path):
        if full_path.suffix != '.ptr':
            raise ValueError (f'Wrong file extension, expected .ptr, got {pathlib.Path(full_path).suffix}')

    def file_exists_check(self, full_path:pathlib.Path):
        if not full_path.is_file():
            raise FileNotFoundError (f'File at path {full_path}')

    def folder_exists_check(self, file_name:pathlib.Path):
        self.save_folder_path =  self.folder_path.joinpath(file_name.stem)
        if not self.save_folder_path.is_dir():
            self.save_folder_path.mkdir()

    def exhaustive_check(self, file_name:pathlib.Path):
        full_path = self.folder_path.joinpath(file_name)
        self.file_extension_check(full_path)
        self.file_exists_check(full_path)
        self.folder_exists_check(file_name)

    def save_npy_file(self, file_name:pathlib.Path, file_id:str):
        data_path = self.save_folder_path.joinpath(f'{file_id}.npy')
        if not data_path.is_file():
            data = self.matlab_bridge.fetch_wrapper_bytes(file_name)
            np.save(data_path, data)

    def process_wrapper_bytes(self, file_name:pathlib.Path):
        self.save_npy_file(file_name, 'wrapper_bytes')

    def process_det_no(self, file_name:pathlib.Path, det_no_key:int):
        self.save_npy_file(file_name, f'det_no_{det_no_key}')

    def process_table_no(self, file_name:pathlib.Path, table_no_key:int):
        self.save_npy_file(file_name, f'table_no_{table_no_key}')

    def get_n_detector_row(self, file_name:pathlib.Path):
        meta_data = json.load(open(f'{file_name.with_suffix("")}/UID_0.json'))
        n_det_row = meta_data['ScanDescr']['Type']['Slices']+1-meta_data['ScanDescr']['Type']['SlicesMon']
        print(f"N Detector Row = {n_det_row}")
        return n_det_row

    def get_n_readings(self, file_name:pathlib.Path):
        meta_data = json.load(open(f'{file_name.with_suffix("")}/UID_0.json'))
        n_readings = meta_data['ScanDescr']['NoOfReadings']
        print(f"N Readings= {n_readings}")
        return n_readings

    def process_UID(self, file_name:pathlib.Path, UID_key:int):
        uid_processor = UIDPROCESSOR(file_name, UID_key)
        UID_data = self.matlab_bridge.fetch_UID(file_name, UID_key)

        if len(UID_data)==0:
            print(f'No entry for UID {UID_key}')

        else:
            if UID_key == 0:
                member_structures = np.dtype(UID_data.dtype).names
                for index, member_structure in enumerate(member_structures):
                    uid_processor.process_member_structure(UID_data[0,0][index], member_structure)

            elif UID_key == 12:
                data, header = UID_data[0], UID_data[1]
                ## First, we save the data as a np file
                metadata_data_file_name = str(self.save_folder_path.joinpath(f'UID_12_data.npy'))
                if not pathlib.Path(metadata_data_file_name).is_file():
                    np.save(metadata_data_file_name, data)
                ## Then, we process the header as we would for a matlab struct
                member_structures = np.dtype(header[0,0].dtype).names
                for index, member_structure in enumerate(member_structures):
                    # Don't ask me what this horror is header[0][0][0][0][index], it's nested format returned from the .mat file
                    uid_processor.process_member_structure(header[0][0][0][0][index], member_structure)

            elif UID_key == 20:
                for key, value in UID_data.items():
                    uid_processor.process_member_structure(value, key)

            elif UID_key == 50:
                for cell in UID_data:
                    try:
                        member_structures = cell.keys()
                        for member_structure in member_structures:
                            uid_processor.process_member_structure(cell[member_structure], member_structure)
                    except AttributeError:
                        print('Empty cell, passing')

            else:
                raise NotImplementedError(f'Returned data of UID {UID_key} never encountered, passing...')

    def process_ptr_file(self, file_name:pathlib.Path):
        self.exhaustive_check(file_name)
        self.matlab_bridge.set_save_folder_path(self.folder_path.joinpath(file_name.stem))
        for UID_key in [0, 12, 20, 30, 50, 60, 61]:
            self.process_UID(file_name, UID_key)
        self.process_wrapper_bytes(file_name)
        self.process_det_no(file_name, 1)
        self.process_det_no(file_name, 2)
        self.process_table_no(file_name, 1)
        self.process_table_no(file_name, 2)
        self.matlab_bridge.fetch_readings(file_name, self.get_n_readings(file_name),self.get_n_detector_row(file_name))

    def process_full_folder(self):
        file_names = self.folder_path.glob('*.ptr')
        for file_name in file_names:
            self.process_ptr_file(file_name)

def process_raw_file(temp_dir_path:pathlib.Path, debug=False):
    print(f'Processing folder {temp_dir_path}')
    if not debug:
        assert temp_dir_path.is_dir()
        ## Create File Processor object
        files_processor = PTRFILEPROCESSOR(temp_dir_path)
        for raw_file_path in temp_dir_path.glob('*.ptr'):
            print(f'Processing file {raw_file_path}')
            assert raw_file_path.suffix =='.ptr'
            files_processor.process_full_folder()
            ## Clean up
            print('Clean up operation: removing .ptr file...')
            raw_file_path.unlink()

if __name__ == '__main__':
    folder_path = r'C:\Users\hx21262\Downloads\TEMP_XNAT\100_20200414\resources\RAW\files'
    file_processor = PTRFILEPROCESSOR(folder_path)
    file_processor.process_ptr_file(pathlib.Path(r'C:\Users\hx21262\Downloads\TEMP_XNAT\100_20200414\resources\RAW\files\66862.Anonymous.CT..601.RAW.20200414.110431.963978.2020.07.03.21.11.10.086000.1309236364.ptr'))



