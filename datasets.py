from typing import List, Tuple, Dict
import pathlib
import random
import math

import torch
import numpy as np
import json
from torch.utils.data import Dataset

from backends.odl import ODLBackend

def format_index(index:int) -> str:
    str_index = str(index)
    while len(str_index) < 4:
        str_index = '0' + str_index
    assert len(str_index) == 4
    return str_index

def load_json(file_path:pathlib.Path):
    if not file_path.is_file():
        raise FileNotFoundError (f'No file found at {file_path}')
    with open(file_path, 'r') as file_read:
        file = json.load(file_read)
    return file

class LIDC_IDRI(Dataset):
    def __init__(
            self,
            path_to_dataset:pathlib.Path,
            pipeline:str,
            backend:ODLBackend,
            training_proportion:float,
            training:bool,
            is_subset: bool,
            transform=None,
            subset=[],
            verbose=False,
            patient_list=[]
            ):

        ## Defining the path to data
        self.path_to_processed_dataset = path_to_dataset
        self.patients_masks_dictionary = load_json(self.path_to_processed_dataset.joinpath('patients_masks.json'))
        self.patients_diagnosis_dictionary = load_json(self.path_to_processed_dataset.joinpath('patient_id_to_diagnosis.json'))

        ## Instanciating the class attributes from constructor argument
        self.pipeline = pipeline
        self.backend = backend
        self.training_proportion = training_proportion
        self.training = training
        ## Subset is either a boolean with False value OR a List of patient Ids
        self.is_subset = is_subset
        self.transform = transform

        ## Partitioning the dataset
        if is_subset == False:
            self.total_patients = 1012
            self.subset = [index for index in range(1,self.total_patients)]
        else:
            self.total_patients = len(subset)
            self.subset = subset

        ### Making a correspondance between patient ID (LIDC-IDRI-xxxx) and the number of slices of the patient
        self.patient_index_to_n_slices_dict = {
            f'LIDC-IDRI-{format_index(index)}' : len(list(self.path_to_processed_dataset.joinpath(f'LIDC-IDRI-{format_index(index)}').glob('slice_*.npy'))) for index in self.subset
        }

        ### Calculating the number of patients
        ### Note: we do sanity checks here with testing and training regardless of the self.mode argument
        self.n_patients_training = math.floor(self.training_proportion*self.total_patients)
        self.n_patients_testing  = math.ceil((1-self.training_proportion)*self.total_patients)
        assert self.total_patients == (self.n_patients_training + self.n_patients_testing), print(
            f'Total patients: {self.total_patients}, \n training patients {self.n_patients_training}, \n testing patients {self.n_patients_testing}'
            )

        self.patient_indexs = list(self.patient_index_to_n_slices_dict.keys())
        if len(patient_list)==0:
            self.training_patients_list = self.patient_indexs[:self.n_patients_training]
            self.testing_patients_list = self.patient_indexs[self.n_patients_training:]
            assert len(self.patient_indexs) == len(self.training_patients_list) + len(self.testing_patients_list), print(
                f'Len patients ids: {len(self.patient_indexs)}, \n len training patients {len(self.training_patients_list)}, \n len testing patients {len(self.testing_patients_list)}'
                )

        else:
            self.testing_patients_list:List[str] = patient_list #type:ignore

        if verbose:
            print('Preparing patient list, this may take time....')

        if self.training:
            self.slice_index_to_patient_index_list = self.get_slice_index_to_patient_index_list(self.training_patients_list)
            self.patient_index_to_first_index_dict = self.get_patient_index_to_first_index_dict(self.training_patients_list)

        else:
            self.slice_index_to_patient_index_list = self.get_slice_index_to_patient_index_list(self.testing_patients_list) #type:ignore
            self.patient_index_to_first_index_dict = self.get_patient_index_to_first_index_dict(self.testing_patients_list) #type:ignore

        if verbose:
            print(f'Patient lists ready')

    def get_patient_slices_list(self, patient_index:str) -> List:
        assert patient_index in self.patient_index_to_n_slices_dict, f'Patient Id {patient_index} not in {self.patient_index_to_n_slices_dict}'
        n_slices = self.patient_index_to_n_slices_dict[patient_index]
        if self.pipeline == 'reconstruction':
            patient_slices_list = []
            for slice_index in range(n_slices):
                patient_slices_list.append(self.path_to_processed_dataset.joinpath(f'{patient_index}/slice_{slice_index}.npy'))

            return patient_slices_list

        else:
            raise NotImplementedError

    def get_patient_index_to_first_index_dict(self, patient_list:List):
        patient_index_to_first_index_dict = {}
        global_index = 0
        for patient_index in patient_list:
            path_to_folder = self.path_to_processed_dataset.joinpath(patient_index)
            patient_index_to_first_index_dict[patient_index] = global_index
            global_index += len(list(path_to_folder.glob('slice_*.npy')))
        return patient_index_to_first_index_dict

    def get_slice_index_to_patient_index_list(self, patient_list:List):
        slice_index_to_patient_index_list = []
        for patient_index in patient_list:
            path_to_folder = self.path_to_processed_dataset.joinpath(patient_index)
            n_slices = len(list(path_to_folder.glob('slice_*.npy')))
            for slice_index in range(n_slices):
                slice_index_to_patient_index_list.append(patient_index)
        return slice_index_to_patient_index_list

    def get_reconstruction_tensor(self, file_path:pathlib.Path) -> torch.Tensor:
        tensor = torch.from_numpy(np.load(file_path)).unsqueeze(0)
        return tensor

    def get_sinogram_tensor(self, file_path:pathlib.Path, backend:ODLBackend) -> torch.Tensor:
        #### EXPENSIVE ####
        return backend.get_sinogram(self.get_reconstruction_tensor(file_path))

    def get_filtered_backprojection(self, file_path:pathlib.Path, backend:ODLBackend) -> torch.Tensor:
        return torch.from_numpy(backend.get_filtered_backprojection(backend.operator(np.load(file_path)), 'Hann')).unsqueeze(0) #type:ignore

    def get_mask_tensor(self, patient_index:str, slice_index:int) -> torch.Tensor:
        mask = torch.zeros((512,512), dtype=torch.bool)
        ## First, assess if the slice has a nodule
        try:
            all_nodules_dict:Dict = self.patients_masks_dictionary[patient_index][f'{slice_index}']
            for nodule_index, nodule_annotations_list in all_nodules_dict.items():
                ## If a nodule was not segmented by all the clinicians, the other annotations should not always be seen
                while len(nodule_annotations_list) < 4:
                    nodule_annotations_list.append('')

                annotation = random.choice(nodule_annotations_list)
                if annotation == '':
                    nodule_mask = torch.zeros((512,512), dtype=torch.bool)
                else:
                    path_to_mask = self.path_to_processed_dataset.joinpath(f'{patient_index}/mask_{slice_index}_nodule_{nodule_index}_annotation_{annotation}.npy')
                    nodule_mask = torch.from_numpy(np.load(path_to_mask))

                mask = mask.bitwise_or(nodule_mask)

        except KeyError:
            pass
        # byte inversion
        mask = mask.int()
        background = 1- mask
        return torch.stack((background, mask))

    def __len__(self):
        return len(self.slice_index_to_patient_index_list)

    def get_patient_slice_index_path(self, patient_index:str, slice_index:int):
        return self.path_to_processed_dataset.joinpath(f'{patient_index}/slice_{slice_index}.npy')

    def get_specific_slice(self, patient_index:str, slice_index:int):
        ## Assumes slice and mask exist
        file_path = self.get_patient_slice_index_path(patient_index, slice_index)
        return self.get_reconstruction_tensor(file_path), self.get_mask_tensor(patient_index, slice_index)

    def __getitem__(self, index):
        patient_index = self.slice_index_to_patient_index_list[index]
        first_slice_index = self.patient_index_to_first_index_dict[patient_index]
        slice_index = index - first_slice_index
        #print(f'Index, {index}, Patient Id : {patient_index}, first_slice_index : {first_slice_index}, slice_index : {slice_index} ')
        file_path = self.path_to_processed_dataset.joinpath(f'{patient_index}/slice_{slice_index}.npy')

        ### WE NEVER RETURN THE SINOGRAM TO AVOID COMPUTING IT PER SAMPLE ###
        ### (except when we want the filtered backprojection...) ###
        if self.pipeline == "joint" or self.pipeline == "end_to_end" or self.pipeline == "segmentation":
            reconstruction_tensor = self.get_reconstruction_tensor(file_path)
            mask_tensor = self.get_mask_tensor(patient_index, slice_index)
            if self.transform is not None:
                reconstruction_tensor = self.transform['reconstruction_transforms'](reconstruction_tensor)
                mask_tensor = self.transform['mask_transforms'](mask_tensor)
            return reconstruction_tensor, mask_tensor

        elif self.pipeline == "reconstruction" or self.pipeline=="fourier_filter":
            reconstruction_tensor = self.get_reconstruction_tensor(file_path)
            #filtered_backprojection = self.get_filtered_backprojection(file_path, self.backend)
            if self.transform is not None:
                reconstruction_tensor = self.transform['reconstruction_transforms'](reconstruction_tensor)
            return reconstruction_tensor

        elif self.pipeline == "diagnostic":
            return self.patients_diagnosis_dictionary[patient_index]

        else:
            raise NotImplementedError

