from typing import List, Dict
from pathlib import Path
import random
import math
import ast

import pandas as pd
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

class PatientDataset(Dataset):
    def __init__(
        self,
        path_to_processed_dataset:Path,
        patient_index:str,
        slices,
        annotations_dataframe:pd.DataFrame,

        transform = None) -> None:
        ## Defining the path to data
        self.path_to_processed_dataset = path_to_processed_dataset
        ###
        self.patient_index = patient_index
        self.slices = ast.literal_eval(slices)
        self.annotations_dataframe = annotations_dataframe
        self.transform=transform
        print(f'The PatientDataset Object for the patient {patient_index} has {len(self.slices)} slices')
        print(f'Of these, there are {len(self.annotations_dataframe)} annotations')

    def compute_mask_tensor(self, patient_index: str, slice_index: int) -> torch.Tensor:
        mask = torch.zeros((512, 512), dtype=torch.bool)
        df_query = f'"{patient_index}" == patient_index & {slice_index} == slice'
        subset = self.annotations_dataframe.query(df_query)
        slice_nodules = subset['nodule'].unique()
        for nodule in slice_nodules:
            annotations = subset[subset['nodule'] == nodule]
            annotation_row = annotations.sample()
            nodule_index = annotation_row['nodule'].values[0]
            annotation_index = annotation_row['annotation'].values[0]
            path_to_mask = self.path_to_processed_dataset.joinpath(
                f"{patient_index}/mask_{slice_index}_nodule_{nodule_index}_annotation_{annotation_index}.npy"
            )
            nodule_mask = torch.from_numpy(np.load(path_to_mask))
            mask = mask.bitwise_or(nodule_mask)
        mask = mask.int()
        return mask.unsqueeze(0)

    def compute_reconstruction_tensor(self, patient_index:str, slice_index:int) -> torch.Tensor:
        return torch.from_numpy(
            np.load(self.path_to_processed_dataset.joinpath(f'{patient_index}/slice_{slice_index}.npy'))
            ).unsqueeze(0)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        ### Compute reconstruction tensor
        reconstruction_tensor = self.compute_reconstruction_tensor(self.patient_index, self.slices[index])
        ### Compute mask tensor
        mask_tensor  = self.compute_mask_tensor(self.patient_index, index)
        if self.transform is not None:
            reconstruction_tensor = self.transform["reconstruction_transforms"](
                reconstruction_tensor
            )
            mask_tensor = self.transform["mask_transforms"](mask_tensor)
        return index, reconstruction_tensor, mask_tensor

class LIDC_IDRI_SEGMENTATIONS(Dataset):
    def __init__(
        self,
        path_to_processed_dataset:Path,
        training_proportion: float,
        training: bool,
        query_string:str,
        transform=None,
        verbose=True
        ):
        ## Defining the path to data
        self.path_to_processed_dataset = path_to_processed_dataset
        """
        The path_to_processed_dataset attribute (pathlib.Path) points towards the folder
        where the data is stored
        """
        ## Defining dataset partition
        self.training_proportion = training_proportion
        """
        The training_proportion attribute (float) indicates what proportion of the
        dataset is used for training
        """
        self.training = training
        """
        The training attribute (bool) set the training/testing mode for the dataset
        """
        ## Defining the transform
        self.transform = transform
        """
        The transform attribute (torch.nn.Compose() | None) specifies what transforms need to
        be applied to the training data when __getitem__ function is called
        """
        ## Defining the annotation dataframe path
        self.annotations_dataframe_path = self.path_to_processed_dataset.joinpath('annotations_dataframe.csv')
        """
        The annotation_dataframe_path (pathlib.Path) points towards the csv dataframe that maps patient indices
        to their metadata, see self.dataframe doc
        """
        ## Get the annotation dataframe
        if not self.annotations_dataframe_path.is_file():
            print('Dataframe file not found, computing...')
            self.compute_annotations_dataset_dataframe()
        self.annotations_dataframe = pd.read_csv(self.annotations_dataframe_path)
        """
        The dataframe (pd.Dataframe) attribute maps \n
        patient_index / slice_index / nodule_index / annotation_index / nodule_size \n
        It is they key element of the dataset, as it can be queried to return only the
        rows that contain the data of interest
        """
        ## Defining the all slices dataframe path
        self.all_slices_dataframe_path = self.path_to_processed_dataset.joinpath('all_slices_dataframe.csv')
        """
        The all_slices_dataframe_path (pathlib.Path) points towards the csv dataframe that maps patient indices
        to their slices
        """
        ## Get the annotation dataframe
        if not self.all_slices_dataframe_path.is_file():
            print('Dataframe file not found, computing...')
            self.compute_all_slices_dataset_dataframe()
        self.all_slices_dataframe = pd.read_csv(self.all_slices_dataframe_path)
        """
        The dataframe (pd.Dataframe) attribute maps \n
        patient_index / slice_index / nodule_index / annotation_index / nodule_size \n
        It is they key element of the dataset, as it can be queried to return only the
        rows that contain the data of interest
        """

        ## Query the subset of the dataframe
        self.annotations_dataframe_subset = self.annotations_dataframe.query(query_string)
        """
        The dataframe_subset (pd.Dataframe) is the subset of the self.dataframe attribute that
        contains only the annotations that comply with the query
        """
        ## Get the unique patient indices
        self.all_patients_lists = self.annotations_dataframe_subset['patient_index'].unique()
        """
        The all_patients_lists (List) attribute is the list of unique patient indices that
        have annotations that comply with the query
        """
        self.total_patients  = len(self.all_patients_lists)
        """
        The total_patients (int) attribute is the number of different patients in the dataset
        """
        ## Partitioning the dataset

        self.patients_list: List
        """
        The patients_list (List) attribute is the list of patient indices that will be used for
        training or testing, depending on the self.training attribute
        <!> Type Checking Error <!>
        --> Type checking will return an error when accessing self.patient_indices as a list,
            as returned type from pd.Series.unique is np.ndarray.
            No error is raised at runtime so it should be alright.
        <!> Attribute Naming <!>
        --> Compared to all_patients_list, patients_list attribute only contains the patients
            used for training or testing, based on the self.training attribute
        """

        if self.training:
            self.patients_list = self.all_patients_lists[ #type:ignore
                :math.floor(self.training_proportion * self.total_patients)
                ]

        else:
            self.patients_list = self.all_patients_lists[ #type:ignore
                math.ceil((self.training_proportion) * self.total_patients) :
                ]

        self.annotations_dataframe_query = self.annotations_dataframe_subset[
            self.annotations_dataframe_subset['patient_index'].isin(self.patients_list)
            ]
        """
        The annotations_dataframe (pd.Dataframe) attribute is the subset of the self.dataframe_subset
        attribute that will be used for training or testing, based on the self.training attribute
        """
        self.patient_to_annotated_slices = self.annotations_dataframe[
            self.annotations_dataframe['patient_index'].isin(self.patients_list)
            ]
        """
        The patient_to_slices (pd.Dataframe) attribute maps the all the slices of each patient
        used for training or testing, based on the self.training attribute
        """

        self.compute_index_dataframe()

        self.index_dataframe:pd.DataFrame
        """
        The index_dataframe (pd.Dataframe) attribute maps patient indices to slices
        that have annotations of interest. It is made from a dict that looks like \n
        {
            patient_index : ['LIDC-IDRI-xxxx', 'LIDC-IDRI-xxxx', ...], \n
            slice_index :   ['y', 'y+1', ...] \n
        }
        """

    def compute_index_dataframe(self):
        index_dict = {
            'patient_index' : [],
            'slice_index' : [],
        }
        for patient_index in self.patients_list:
            patient_annotations = self.annotations_dataframe.query(f'"{patient_index}" == patient_index')
            unique_slices = patient_annotations['slice'].unique()
            for slice_index in unique_slices:
                index_dict['patient_index'].append(patient_index)
                index_dict['slice_index'].append(slice_index)
        self.index_dataframe = pd.DataFrame.from_dict(index_dict)

    def get_all_patient_slices(self, patient_index:str) -> List[int]:
        return self.all_slices_dataframe[self.all_slices_dataframe['patient_index'] == patient_index]['slices'].iloc[0]

    def compute_all_slices_dataset_dataframe(self):
        print('Computing all slices dataframe...')
        slices_dict = {
            'patient_index' : [],
            'slices': []
        }
        for patient_index in range(1, 1012):
            patient_index = f'LIDC-IDRI-{format_index(patient_index)}'
            path_to_patient = self.path_to_processed_dataset.joinpath(patient_index)
            n_slices = len(list(path_to_patient.glob(f'slice_*')))
            slices_dict['patient_index'].append(patient_index)
            slices_dict['slices'].append([i for i in range(n_slices)])
        dataframe = pd.DataFrame.from_dict(slices_dict)
        dataframe.to_csv(self.all_slices_dataframe_path, index=False)

    def compute_annotations_dataset_dataframe(self):
        print('Computing all annotations dataframe...')
        annotation_dict = {
        'patient_index' : [],
        'slice': [],
        'slice_path':[],
        'nodule' : [],
        'annotation' : [],
        'nodule_size':[]
        }

        for patient_index in range(1, 1012):
            patient_index = f'LIDC-IDRI-{format_index(patient_index)}'
            path_to_patient = self.path_to_processed_dataset.joinpath(patient_index)
            for mask_file_path in list(path_to_patient.glob(f'mask_*')):
                slice_number  = mask_file_path.stem.split('_')[1]
                nodule_number = mask_file_path.stem.split('_')[3]
                annotation_number = mask_file_path.stem.split('_')[5]

                mask = np.load(mask_file_path)
                n_non_zeros = len(np.nonzero(mask)[0])

                annotation_dict['patient_index'].append(patient_index)
                annotation_dict['slice'].append(slice_number)
                annotation_dict['slice_path'].append(f'{patient_index}/{mask_file_path.name}')
                annotation_dict['nodule'].append(nodule_number)
                annotation_dict['annotation'].append(annotation_number)
                annotation_dict['nodule_size'].append(n_non_zeros)

        dataframe = pd.DataFrame.from_dict(annotation_dict)
        dataframe.to_csv(self.annotations_dataframe_path, index=False)

    def get_patient_dataset(self, patient_index:str) -> PatientDataset:
        return PatientDataset(
            self.path_to_processed_dataset,
            patient_index,
            self.get_all_patient_slices(patient_index),
            annotations_dataframe=self.annotations_dataframe_query[self.annotations_dataframe_query['patient_index'] == patient_index],
            transform=self.transform
        )

    def compute_mask_tensor(self, patient_index: str, slice_index: int) -> torch.Tensor:
        mask = torch.zeros((512, 512), dtype=torch.bool)
        df_query = f'"{patient_index}" == patient_index & {slice_index} == slice'
        subset = self.annotations_dataframe_query.query(df_query)
        slice_nodules = subset['nodule'].unique()
        for nodule in slice_nodules:
            annotations = subset[subset['nodule'] == nodule]
            annotation_row = annotations.sample()
            nodule_index = annotation_row['nodule'].values[0]
            annotation_index = annotation_row['annotation'].values[0]
            path_to_mask = self.path_to_processed_dataset.joinpath(
                f"{patient_index}/mask_{slice_index}_nodule_{nodule_index}_annotation_{annotation_index}.npy"
            )
            nodule_mask = torch.from_numpy(np.load(path_to_mask))
            mask = mask.bitwise_or(nodule_mask)
        mask = mask.int()
        return mask.unsqueeze(0)

    def compute_reconstruction_tensor(self, patient_index:str, slice_index:int) -> torch.Tensor:
        return torch.from_numpy(
            np.load(self.path_to_processed_dataset.joinpath(f'{patient_index}/slice_{slice_index}.npy'))
            ).unsqueeze(0)

    def __len__(self):
        return len(self.index_dataframe)

    def __getitem__(self, index):
        ## Access row
        row = self.index_dataframe.iloc[index]
        ## Unpack row information
        patient_index = row['patient_index']
        slice_index = row['slice_index']
        ### Compute reconstruction tensor
        reconstruction_tensor = self.compute_reconstruction_tensor(patient_index, slice_index)
        ### Compute mask tensor
        mask_tensor  = self.compute_mask_tensor(patient_index, slice_index)
        if self.transform is not None:
            reconstruction_tensor = self.transform["reconstruction_transforms"](
                reconstruction_tensor
            )
            mask_tensor = self.transform["mask_transforms"](mask_tensor)
        return reconstruction_tensor, mask_tensor

class LIDC_IDRI(Dataset):
    def __init__(
        self,
        path_to_dataset: Path,
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
        self.patient_id_to_all_slices: Dict = load_json(patient_index_to_n_slices_file_path)
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

            self.patients_masks_all = load_json(self.path_to_processed_dataset.joinpath("patients_masks.json"))

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

    def compute_patient_index_to_segmented_slices(self, save_file_path: Path):
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

    def compute_patient_index_to_slices(self, save_file_path: Path):
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

    def get_reconstruction_tensor(self, file_path: Path) -> torch.Tensor:
        tensor = torch.from_numpy(np.load(file_path)).unsqueeze(0)
        return tensor

    def get_sinogram_tensor(self, file_path: Path) -> torch.Tensor:
        #### EXPENSIVE ####
        return self.backend.get_sinogram(self.get_reconstruction_tensor(file_path))

    def get_filtered_backprojection(self, file_path: Path) -> torch.Tensor:
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
