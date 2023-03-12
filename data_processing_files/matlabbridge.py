import scipy
import matlab.engine
import numpy as np
from tqdm import tqdm
import pathlib

from constants import MATLAB_FUNCS_DIR
from upload_utils import PARAMIKO_PROTOCOL

## The MATLABBRIDGE class is an interface between the Python script and the Matlab engine that is *required* to read the proprietary Siemens data format
# It requires the path of the folder in which there are the files we want to extract, and (implicitly) the folder where the said matlab functions are stored
# Whilst some functions are redundant, I (Emilien Valat) chose to have a very explicit factory design pattern for better readbility.
class MATLABBRIDGE:
    def __init__(self, folder_path:pathlib.Path) -> None:
        print(f'Instanciate MATLABBRIDGE object for folder {folder_path}')
        self.folder_path = folder_path
        self.eng = None
        self.eng = matlab.engine.start_matlab()
        self.eng.cd(str(self.folder_path), nargout=0) # type:ignore
        self.eng.addpath(self.eng.genpath(str(MATLAB_FUNCS_DIR)), nargout=0) # type:ignore
        self.current_ptr_file_path = None

    def set_current_ptr_file_path(self, current_ptr_file_path:pathlib.Path):
        self.current_ptr_file_path = current_ptr_file_path

    def set_save_folder_path(self, save_folder_path:pathlib.Path):
        self.save_folder_path = save_folder_path

    def fetch_image(self, image_no:int) -> np.ndarray:
        return self.eng.read_xacb2(str(self.current_ptr_file_path), 'ImageNo', float(image_no))[0] # type:ignore

    def fetch_scan(self, scan_no:int) -> np.ndarray:
        return self.eng.read_xacb2(str(self.current_ptr_file_path), 'ScanNo', float(scan_no))[0] # type:ignore

    def fetch_reading(self,scan_no:int, reading_no:int) -> np.ndarray:
        return self.eng.read_xacb2(str(self.current_ptr_file_path), 'ReadingNo', float(reading_no), 'ScanNo', float(scan_no))[0] # type:ignore

    def fetch_header(self, header_save_path:pathlib.Path):
        self.eng.read_header(str(self.current_ptr_file_path), str(header_save_path), nargout=0) # type:ignore

    def fetch_scans(self, n_readings:int):
        raise NotImplementedError

    def fetch_images(self, n_readings:int):
        raise NotImplementedError

    def fetch_readings(self, n_readings:int, n_detector_rows:int, sftp_protocol:PARAMIKO_PROTOCOL):
        sinogram = np.zeros((736, n_readings*n_detector_rows))
        for scan_no in range(1,2):
            for reading_no in tqdm(range(1, 1+n_readings)):
                sinogram[:,(reading_no-1)*n_detector_rows:reading_no*n_detector_rows] = self.fetch_reading(scan_no, reading_no)
        local_save_path  = self.save_folder_path.joinpath('sinogram.npy')
        remote_save_path = sftp_protocol.remote_dir_path + '/sinogram.npy'
        sftp_protocol.save_upload(sinogram, local_save_path, remote_save_path)

    def mat_file_temp_save(self, uid_key:int):
        metadata_data_file_name = self.save_folder_path.joinpath(f'UID_{uid_key}.mat')
        if not pathlib.Path(metadata_data_file_name).is_file():
            print(f'No file at {metadata_data_file_name}, fetching from .ptr file and saving to .mat file ...')
            self.eng.read_UID(str(self.current_ptr_file_path), str(metadata_data_file_name), uid_key, nargout=0) # type:ignore
        print(f'Reading meta data file at {metadata_data_file_name}')
        return scipy.io.loadmat(metadata_data_file_name)['UID_data']

    def fetch_UID_0(self) -> np.ndarray:
        ## Here, we have to save a temp .mat file as we get the error: "ValueError: only a scalar struct can be returned from MATLAB"
        return self.mat_file_temp_save(0)

    def fetch_UID_12(self):
        ## Here, we have to save a temp .mat file as we get the error: "ValueError: only a scalar struct can be returned from MATLAB"
        ## Having the default nargout = 1 will return the data part of UID 12 only
        data   = self.eng.return_UID(str(self.current_ptr_file_path), 12, nargout = 1) # type:ignore
        header = self.mat_file_temp_save(12)
        return data, header

    def fetch_UID_20(self) -> np.ndarray:
        return self.eng.return_UID(str(self.current_ptr_file_path), 20, nargout = 1) # type:ignore

    def fetch_UID_30(self) -> np.ndarray:
        return self.eng.return_UID(str(self.current_ptr_file_path), 30, nargout = 1) # type:ignore

    def fetch_UID_50(self) -> np.ndarray:
        return self.eng.return_UID(str(self.current_ptr_file_path), 50, nargout = 1) # type:ignore

    def fetch_UID_60(self) -> np.ndarray:
        return self.eng.return_UID(str(self.current_ptr_file_path), 60, nargout = 1) # type:ignore

    def fetch_UID_61(self) -> np.ndarray:
        return self.eng.return_UID(str(self.current_ptr_file_path), 61, nargout = 1) # type:ignore

    def fetch_UID(self, UID_key:int) -> np.ndarray:
        if UID_key == 0:
            return self.fetch_UID_0()
        elif UID_key == 12:
            return self.fetch_UID_12()# type:ignore
        elif UID_key == 20:
            return self.fetch_UID_20()
        elif UID_key == 30:
            return self.fetch_UID_30()
        elif UID_key == 50:
            return self.fetch_UID_50()
        elif UID_key == 60:
            return self.fetch_UID_60()
        elif UID_key == 61:
            return self.fetch_UID_61()
        else:
            raise NotImplementedError(f'Fetch function not implemented for {UID_key}')

    def fetch_wrapper_bytes(self):
        print(f'Getting wrapper bytes for file {self.current_ptr_file_path}')
        return self.eng.read_xacb2(str(self.current_ptr_file_path), 'wrapperBytes', 0.0)[0] # type:ignore

    def fetch_det_no(self, det_no_key:int):
        print(f'Getting det {det_no_key} for file {self.current_ptr_file_path}')
        return self.eng.read_xacb2(str(self.current_ptr_file_path), 'DetNo', float(det_no_key))[0] # type:ignore

    def fetch_table_no(self, table_no_key:int):
        print(f'Getting table {table_no_key} for file {self.current_ptr_file_path}')
        return self.eng.read_xacb2(str(self.current_ptr_file_path), 'TableNo', float(table_no_key))[0] # type:ignore

    def fetch_n_readings(self) -> int:
        print(f'Getting number of readings for file {self.current_ptr_file_path}')
        return int(self.eng.get_n_readings(str(self.current_ptr_file_path), nargout = 1)) # type:ignore