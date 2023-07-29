from typing import List, Dict
import pathlib
import random
import math

import torch
import numpy as np
from torch.utils.data import Dataset

from utils import load_json, save_json
from backends.odl import ODLBackend

def format_index(index: int) -> str:
    str_index = str(index)
    while len(str_index) < 4:
        str_index = "0" + str_index
    assert len(str_index) == 4
    return str_index


class LIDC_IDRI(Dataset):
    def __init__(
        self,
        path_to_dataset: pathlib.Path,
        pipeline: str,
        backend: ODLBackend,
        training_proportion: float,
        training: bool,
        is_subset: bool,
        transform=None,
        subset=[],
        verbose=True,
        patient_list=[],
        rule='superior',
        subtelty_value=4,
        annotation_size=512
        ):
        ## Instanciating the class attributes from constructor argument
        self.pipeline = pipeline
        self.backend = backend
        self.training_proportion = training_proportion
        self.training = training
        ## Subset is either a boolean with False value OR a List of patient Ids
        self.is_subset = is_subset
        if self.is_subset:
            assert len(subset) != 0, f'No subset List argument provided, please provide the patient indices as int values. ex: [3,100,930,...]'
        self.transform = transform
        ## Defining the path to data
        self.path_to_processed_dataset = path_to_dataset

        ## Define path to patient_id_to_diagnosis
        """
        "LIDC-IDRI-xxxx": diagnostic_key (cf tcia-diagnosis-data-2012-04-20.xls)
        """
        self.patient_id_to_diagnosis = load_json(self.path_to_processed_dataset.joinpath("patient_id_to_diagnosis.json"))

        ## Preprocess the dataset if the file doesn't exists:
        # Patient index to n_slices
        """
        "LIDC-IDRI-xxxx": List of slices : List
        """
        patient_index_to_n_slices_file_path = self.path_to_processed_dataset.joinpath("patient_index_to_slices.json")
        if not patient_index_to_n_slices_file_path.is_file():
            self.compute_patient_index_to_slices(patient_index_to_n_slices_file_path)
        self.patient_id_to_slices_of_interest: Dict = load_json(patient_index_to_n_slices_file_path)

        if pipeline in ['joint', 'segmentation']:
            ## Define path to patients_masks
            """
            "LIDC-IDRI-xxxx": {
                "segmented_slice_index": {
                    "nodule_index": [
                        annotation_i,
                        annotation_j,
                        annotation_k
                    ]
                },
            """
            self.patients_masks = load_json(self.path_to_processed_dataset.joinpath("patients_masks.json"))
            # Patient index to segmented slices indexes
            """
            "LIDC-IDRI-xxxx": segmented slices : List
            """
            patient_index_to_segmented_slices_file_path = (self.path_to_processed_dataset.joinpath("patient_index_to_segmented_slices.json"))
            if not patient_index_to_segmented_slices_file_path.is_file():
                self.compute_patient_index_to_segmented_slices(patient_index_to_segmented_slices_file_path)
            self.patient_index_to_segmented_slices: Dict = load_json(patient_index_to_segmented_slices_file_path)

            ## Patient Indices that have relevant nodules
            """
            ["LIDC-IDRI-xxxx", "LIDC-IDRI-yyyy", "LIDC-IDRI-zzzz"]
            """
            self.patients_with_nodules_of_interest = load_json(self.path_to_processed_dataset.joinpath(f"patients_with_nodule_subtlety_{rule}_to_{subtelty_value}.json"))
            ## Define patient index to segmented slices index
            # We overwite the self.patient_id_to_slices_of_interest attribute with the dict mapping patient name
            self.patient_id_to_slices_of_interest = dict(
                (k, self.patient_index_to_segmented_slices[k]) for k in self.patients_with_nodules_of_interest
            )

            path_to_annotations_file = self.path_to_processed_dataset.joinpath(f"patient_index_to_annotations_superior_to_{annotation_size}.json")
            if not path_to_annotations_file.is_file():
                self.compute_annotation_size(annotation_size)
            self.patient_id_to_slices_of_interest = load_json(path_to_annotations_file)
            print(f'There are {len(self.patient_id_to_slices_of_interest)} patients')
            self.patients_masks = load_json(self.path_to_processed_dataset.joinpath(f"patient_index_to_annotations_superior_to_{annotation_size}.json"))

        ## Partitioning the dataset
        if is_subset == False:
            self.subset = list(self.patient_id_to_slices_of_interest.keys())
        else:
            self.subset = subset

        self.total_patients = len(self.subset)

        ### Calculating the number of patients
        ### Note: we do sanity checks here with testing and training regardless of the self.mode argument
        self.n_patients_training = math.floor(self.training_proportion * self.total_patients)
        self.n_patients_testing = math.ceil((1 - self.training_proportion) * self.total_patients)
        assert self.total_patients == (self.n_patients_training + self.n_patients_testing), \
            print(f"Total patients: {self.total_patients}, \
                \n training patients {self.n_patients_training}, \
                \n testing patients {self.n_patients_testing}")

        self.patient_indices = list(self.patient_id_to_slices_of_interest.keys())

        ### If the patient list is not specified, it means that we are training, not testing
        if len(patient_list) == 0:
            self.training_patients_list = self.patient_indices[:self.n_patients_training]
            self.testing_patients_list = self.patient_indices[self.n_patients_training :]
            assert len(self.patient_indices) == len(self.training_patients_list) + len(self.testing_patients_list), \
                print(f"Len patients ids: {len(self.patient_indices)}, \
                    \n len training patients {len(self.training_patients_list)}, \
                    \n len testing patients {len(self.testing_patients_list)}")

        else:
            self.testing_patients_list: List[str] = patient_list  # type:ignore

        if verbose:
            print("Preparing patient list, this may take time....")

        if self.training:
            self.slice_index_to_patient_index_list, self.patient_index_to_first_index_dict = self.get_dataset_indices(
                self.training_patients_list, self.patient_id_to_slices_of_interest
                )

        else:
            self.slice_index_to_patient_index_list, self.patient_index_to_first_index_dict = self.get_dataset_indices(
                self.testing_patients_list, self.patient_id_to_slices_of_interest
                )


        if verbose:
            print(f"Patient lists ready")

    def compute_patient_index_to_segmented_slices(self, save_file_path: pathlib.Path):
        patient_index_to_segmented_slices = {}
        for index in range(1, 1012):
            patient_name = f"LIDC-IDRI-{format_index(index)}"

            try:
                slices_with_segmentation = list(
                    self.patients_masks[patient_name].keys()
                )
                patient_index_to_segmented_slices[
                    patient_name
                ] = slices_with_segmentation

            except KeyError:
                pass

        save_json(save_file_path, patient_index_to_segmented_slices)

    def compute_patient_index_to_slices(self, save_file_path: pathlib.Path):
        patient_index_to_segmented_slices = {}
        for index in range(1, 1012):
            patient_name = f"LIDC-IDRI-{format_index(index)}"
            path_to_folder = self.path_to_processed_dataset.joinpath(patient_name)
            n_slices = len(list(path_to_folder.glob("slice_*.npy")))
            patient_index_to_segmented_slices[patient_name] = [i for i in range(n_slices)]

        save_json(save_file_path, patient_index_to_segmented_slices)

    def get_dataset_indices(self, patient_list:List, patient_index_to_slices:Dict):
        patient_index_to_first_index_dict = {}
        slice_index_to_patient_index_list = []
        global_index = 0
        for patient_index in patient_list:
            patient_index_to_first_index_dict[patient_index] = global_index
            slice_list = patient_index_to_slices[patient_index]
            global_index += len(patient_index_to_slices[patient_index])
            for _ in slice_list:
                    slice_index_to_patient_index_list.append(patient_index)

        return slice_index_to_patient_index_list, patient_index_to_first_index_dict

    def get_patient_annotations(self, patient_index:str, annotation_size:int) -> Dict:
        path_to_patient = self.path_to_processed_dataset.joinpath(patient_index)
        large_annotations = {}
        for mask_file_path in list(path_to_patient.glob(f'mask_*')):
            slice_number  = mask_file_path.stem.split('_')[1]
            nodule_number = mask_file_path.stem.split('_')[3]
            annotation_number = mask_file_path.stem.split('_')[5]

            mask = np.load(mask_file_path)
            n_non_zeros = len(np.nonzero(mask)[0])

            if annotation_size <= n_non_zeros:
                if slice_number not in large_annotations:
                    large_annotations[slice_number] = {}

                if nodule_number not in large_annotations[slice_number]:
                    large_annotations[slice_number][nodule_number] = []

                large_annotations[slice_number][nodule_number].append(annotation_number)

        return large_annotations

    def compute_annotation_size(self, annotation_size:int) -> None:
        save_file_path = self.path_to_processed_dataset.joinpath(f"patient_index_to_annotations_superior_to_{annotation_size}.json")
        dataset_large_annotations = {}
        for patient_index in range(1, 1012):
            patient_index = f'LIDC-IDRI-{format_index(patient_index)}'
            patient_large_annotations = self.get_patient_annotations(patient_index, annotation_size)
            if len(patient_large_annotations) != 0:
                print(f'Patient {patient_index} has {len(patient_large_annotations)} large annotations')
                dataset_large_annotations[patient_index] = patient_large_annotations

        save_json(save_file_path, dataset_large_annotations)

    def get_reconstruction_tensor(self, file_path: pathlib.Path) -> torch.Tensor:
        tensor = torch.from_numpy(np.load(file_path)).unsqueeze(0)
        return tensor

    def get_sinogram_tensor(self, file_path: pathlib.Path) -> torch.Tensor:
        #### EXPENSIVE ####
        return self.backend.get_sinogram(self.get_reconstruction_tensor(file_path))

    def get_filtered_backprojection(self, file_path: pathlib.Path) -> torch.Tensor:
        return torch.from_numpy(
            self.backend.get_filtered_backprojection(
                self.backend.operator(np.load(file_path)), "Hann" #type:ignore
            )
        ).unsqueeze(0)  # type:ignore

    def get_mask_tensor(self, patient_index: str, slice_index: int) -> torch.Tensor:
        mask = torch.zeros((512, 512), dtype=torch.bool)
        ## First, assess if the slice has a nodule
        all_nodules_dict: Dict = self.patients_masks[patient_index][
            f"{slice_index}"
        ]

        for nodule_index, nodule_annotations_list in all_nodules_dict.items():
            ## If a nodule was not segmented by all the clinicians, the other annotations should not always be seen
            """while len(nodule_annotations_list) < 4:
                nodule_annotations_list.append('')

            annotation = random.choice(nodule_annotations_list)
            if annotation == '':
                nodule_mask = torch.zeros((512,512), dtype=torch.bool)
            else:
                path_to_mask = self.path_to_processed_dataset.joinpath(f'{patient_index}/mask_{slice_index}_nodule_{nodule_index}_annotation_{annotation}.npy')
                nodule_mask = torch.from_numpy(np.load(path_to_mask))
            """

            annotation = random.choice(nodule_annotations_list)
            path_to_mask = self.path_to_processed_dataset.joinpath(
                f"{patient_index}/mask_{slice_index}_nodule_{nodule_index}_annotation_{annotation}.npy"
            )
            nodule_mask = torch.from_numpy(np.load(path_to_mask))

            mask = mask.bitwise_or(nodule_mask)

        # byte inversion
        mask = mask.int()
        background = 1 - mask
        return torch.stack((background, mask))

    def __len__(self):
        return len(self.slice_index_to_patient_index_list)

    def get_patient_slice_index_path(self, patient_index: str, slice_index: int):
        return self.path_to_processed_dataset.joinpath(
            f"{patient_index}/slice_{slice_index}.npy"
        )

    def get_specific_slice(self, patient_index: str, slice_index: int):
        ## Assumes slice and mask exist
        file_path = self.get_patient_slice_index_path(patient_index, slice_index)
        return self.get_reconstruction_tensor(file_path), self.get_mask_tensor(
            patient_index, slice_index
        )

    def __getitem__(self, index):
        patient_index = self.slice_index_to_patient_index_list[index]
        first_slice_index = self.patient_index_to_first_index_dict[patient_index]
        slice_index = index - first_slice_index
        # print(f'Index, {index}, Patient Id : {patient_index}, first_slice_index : {first_slice_index}, slice_index : {slice_index} ')
        if self.pipeline == "reconstruction":
            file_path = self.path_to_processed_dataset.joinpath(
                f"{patient_index}/slice_{slice_index}.npy"
            )
        else:
            slice_name = list(self.patients_masks[patient_index].keys())[
                        slice_index
                    ]


            file_path = self.path_to_processed_dataset.joinpath(
                f"{patient_index}/slice_{slice_name}.npy"
            )
        ### WE NEVER RETURN THE SINOGRAM TO AVOID COMPUTING IT PER SAMPLE ###
        ### (except when we want the filtered backprojection...) ###
        if (
            self.pipeline == "joint"
            or self.pipeline == "end_to_end"
            or self.pipeline == "segmentation"
        ):
            reconstruction_tensor = self.get_reconstruction_tensor(file_path)
            mask_tensor = self.get_mask_tensor(patient_index, slice_name)  # type:ignore
            if self.transform is not None:
                reconstruction_tensor = self.transform["reconstruction_transforms"](
                    reconstruction_tensor
                )
                mask_tensor = self.transform["mask_transforms"](mask_tensor)
            return reconstruction_tensor, mask_tensor

        elif self.pipeline == "reconstruction" or self.pipeline == "fourier_filter":
            reconstruction_tensor = self.get_reconstruction_tensor(file_path)
            # filtered_backprojection = self.get_filtered_backprojection(file_path, self.backend)
            if self.transform is not None:
                reconstruction_tensor = self.transform["reconstruction_transforms"](
                    reconstruction_tensor
                )
            if self.training:
                return reconstruction_tensor
            else:
                return reconstruction_tensor, slice_index

        elif self.pipeline == "diagnostic":
            return self.patient_id_to_diagnosis[patient_index]

        else:
            raise NotImplementedError
