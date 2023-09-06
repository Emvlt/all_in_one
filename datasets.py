from typing import List
from pathlib import Path
import math
import ast

import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset

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
        pipeline:str,
        patient_index:str,
        slices,
        annotations_dataframe=None,
        transform = None) -> None:
        ## Defining the path to data
        self.path_to_processed_dataset = path_to_processed_dataset
        '''
        The path_to_processed_dataset (pathlib.Path) defines the path to the folder where the dataset is stored
        '''
        ### Defining the pipeline
        self.pipeline =pipeline
        '''
        The pipeline (str) defines the pipeline for which the dataset will be used
        '''
        ### Defining the patient index
        self.patient_index = patient_index
        """
        The patient_index (str) is the current patient identifier
        """
        ### Defining the slices of interest
        self.slices = ast.literal_eval(slices)
        """
        The slices (list) defines the list of slice indices of interest
        """
        print(f'The PatientDataset Object for the patient {patient_index} has {len(self.slices)} slices')
        ### Defining the annotations dataframe
        self.annotations_dataframe = annotations_dataframe
        '''
        The annotations_dataframe (pd.Dataframe) holds the relation between
        the patient index, slice index, annotations indices
        '''
        if self.pipeline in ['joint', 'segmentation']:
            print(f'Of these, there are {len(self.annotations_dataframe)} annotations') #type:ignore
        self.transform=transform

    def __str__(self) -> str:
        print_str = f'''
        Pipeline: {self.pipeline}
        Patient index: {self.patient_index}
        Slices: {self.slices}
        '''
        if self.annotations_dataframe is not None:
            print_str += f'''Annotations dataframe:
            {self.annotations_dataframe}
            '''
        else:
            print_str += 'There is no annoation dataframe'
        return print_str

    def compute_mask_tensor(self, slice_index: int) -> torch.Tensor:
        mask = torch.zeros((512, 512), dtype=torch.bool)
        subset = self.annotations_dataframe.query(f'{slice_index} == slice_index') #type:ignore
        slice_nodules = subset['nodule'].unique()
        for nodule in slice_nodules:
            annotations = subset[subset['nodule'] == nodule]
            annotation_row = annotations.sample()
            nodule_index = annotation_row['nodule'].values[0]
            annotation_index = annotation_row['annotation'].values[0]
            path_to_mask = self.path_to_processed_dataset.joinpath(
                f"{self.patient_index}/mask_{slice_index}_nodule_{nodule_index}_annotation_{annotation_index}.npy"
            )
            nodule_mask = torch.from_numpy(np.load(path_to_mask))
            mask = mask.bitwise_or(nodule_mask)
        mask = mask.int()
        return mask.unsqueeze(0)

    def compute_reconstruction_tensor(self, slice_index:int) -> torch.Tensor:
        return torch.from_numpy(
            np.load(self.path_to_processed_dataset.joinpath(f'{self.patient_index}/slice_{slice_index}.npy'))
            ).unsqueeze(0)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        ### Compute reconstruction tensor
        reconstruction_tensor = self.compute_reconstruction_tensor(self.slices[index])
        if self.transform is not None:
            reconstruction_tensor = self.transform["reconstruction_transforms"](
                reconstruction_tensor
            )
        if self.pipeline == 'reconstruction':
            return index, reconstruction_tensor
        else:
            ### Compute mask tensor
            mask_tensor  = self.compute_mask_tensor(index)
            if self.transform is not None:
                mask_tensor = self.transform["mask_transforms"](
                    mask_tensor
                    )
            return index, reconstruction_tensor, mask_tensor

class LIDC_IDRI(Dataset):
    def __init__(
        self,
        path_to_processed_dataset:Path,
        training_proportion: float,
        training: bool,
        pipeline:str,
        query_string=None,
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
        The training attribute (bool) sets the training/testing mode for the dataset
        """
        ## Defining the pipeline
        self.pipeline = pipeline
        """
        The pipeline attribute (str) defines the pipeline for which the dataset will be used
        """
        assert pipeline in ['reconstruction', 'segmentation', 'joint'], 'Wrong pipeline argument'
        ## Defining the transform
        self.transform = transform
        """
        The transform attribute (torch.nn.Compose() | None) specifies what transforms need to
        be applied to the training data when __getitem__ function is called
        """
        if pipeline in ['segmentation', 'joint']:
            assert query_string is not None, 'Specify a value for the query string'
            ## Defining the dataframe path
            self.complete_dataframe_path = self.path_to_processed_dataset.joinpath('complete_dataframe.csv')
            """
            The complete_dataframe_path (pathlib.Path) points towards the csv dataframe that maps patient indices
            to their slice data:
            patient_index | slice_index | slice_path | nodule | nodule_index | annotation_index | annotation_size
            <!> This dataset cannot be iterated upon as is because a same slice can be found n times (if it has n annotations)
            <!> This dataset will be queried and then used for iteration for the segmentation
            """
            ## Get the complete dataframe
            if not self.complete_dataframe_path.is_file():
                print('Dataframe file not found, computing...')
                self.compute_complete_dataframe()
            self.complete_dataframe = pd.read_csv(self.complete_dataframe_path)
            """
            The complete_dataframe (pd.Dataframe) attribute maps \n
            patient_index | slice_index | slice_path | nodule | nodule_index | annotation_index | annotation_size
            It is they key element of the dataset, as it can be queried to return only the
            rows that contain the data of interest
            <!> This dataset cannot be iterated upon as is because a same slice can be found n times (if it has n annotations)
            <!> This dataset will be queried and then used for iteration for the segmentation
            """
            ## Query the subset of the dataframe
            self.dataframe_subset = self.complete_dataframe.query(query_string)
            """
            The dataframe_subset (pd.Dataframe) is the subset of the self.dataframe attribute that
            contains only the annotations that comply with the query
            """
        elif pipeline in ['reconstruction']:
            ### Defining the slices dataframe path
            self.slices_dataframe_path = self.path_to_processed_dataset.joinpath('slices_dataframe.csv')
            """
            The slices_dataframe_path (pathlib.Path) points towards the csv dataframe that maps patient indices
            to their slices data:
            patient_index | slice_index
            <!> This dataset will be used for iteration for the reconstruction
            """
            ## Get the complete dataframe
            if not self.slices_dataframe_path.is_file():
                print('Dataframe file not found, computing...')
                self.compute_slices_dataframe()
            self.dataframe_subset = pd.read_csv(self.slices_dataframe_path)
            """
            The dataframe_subset (pd.Dataframe) attribute maps \n
            patient_index | slice_index
            <!> This dataset will be used for iteration for the reconstruction
            <!> Naming problem:
                -> in order to query simplify the __getitem__ calls, for reconstruction only, we name
            slices_dataframe as dataframe_subset (the subset is, in this case, the whole slices dataframe)
            """
        else:
            raise NotImplementedError

        ## Get the unique patient indices
        self.all_patients_lists = self.dataframe_subset['patient_index'].unique()
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

        self.current_dataframe = self.dataframe_subset[
            self.dataframe_subset['patient_index'].isin(self.patients_list)
            ]
        """
        The current_dataframe (pd.Dataframe) attribute is the subset of the self.dataframe_subset
        attribute that will be used for training or testing, based on the self.training attribute
        It is the dataframe that only holds the patient indices of interest
        """
        self.index_dataframe:pd.DataFrame
        """
        The index_dataframe (pd.Dataframe) attribute maps patient indices to slices
        that have annotations of interest. It is made from a dict that looks like \n
        {
            patient_index : ['LIDC-IDRI-xxxx', 'LIDC-IDRI-xxxx', ...], \n
            slice_index :   ['y', 'y+1', ...] \n
        }
        """
        self.compute_index_dataframe()

    def __str__(self) -> str:
        return f"{self.current_dataframe}"

    def compute_complete_dataframe(self):
        print('Computing dataset dataframe...')
        dataframe_dict = {
        'patient_index' : [],
        'slice_index': [],
        'slice_path':[],
        'nodule' : [],
        'nodule_index' : [],
        'annotation_index' : [],
        'annotation_size':[]
        }

        for patient_index in range(1, 1012):
            patient_index = f'LIDC-IDRI-{format_index(patient_index)}'
            print(f'Processing patient {patient_index}')
            path_to_patient = self.path_to_processed_dataset.joinpath(patient_index)
            n_slices = len(list(path_to_patient.glob(f'slice_*.npy')))
            for slice_index in range(n_slices):

                masks_file_path = list(path_to_patient.glob(f'mask_{slice_index}_*'))
                if len(masks_file_path) == 0:
                    dataframe_dict['patient_index'].append(patient_index)
                    dataframe_dict['slice_index'].append(slice_index)
                    dataframe_dict['slice_path'].append(f'{patient_index}/slice_{slice_index}.npy')
                    dataframe_dict['nodule'].append(False)
                    dataframe_dict['nodule_index'].append(math.nan)
                    dataframe_dict['annotation_index'].append(math.nan)
                    dataframe_dict['annotation_size'].append(math.nan)
                else:
                    for annotation_file_path in masks_file_path:
                        nodule_index = int(annotation_file_path.stem.split('_')[3])
                        annotation_index = int(annotation_file_path.stem.split('_')[5])

                        annotation = np.load(annotation_file_path)
                        annotation_size = len(np.nonzero(annotation)[0])

                        dataframe_dict['patient_index'].append(patient_index)
                        dataframe_dict['slice_index'].append(slice_index)
                        dataframe_dict['slice_path'].append(f'{patient_index}/slice_{slice_index}.npy')
                        dataframe_dict['nodule'].append(True)
                        dataframe_dict['nodule_index'].append(nodule_index)
                        dataframe_dict['annotation_index'].append(annotation_index)
                        dataframe_dict['annotation_size'].append(annotation_size)

        dataframe = pd.DataFrame.from_dict(dataframe_dict)
        dataframe.to_csv(self.complete_dataframe_path, index=False)

    def compute_index_dataframe(self):
        index_dict = {
            'patient_index' : [],
            'slice_index' : [],
        }
        for patient_index in self.patients_list:
            patient_slices = self.current_dataframe.query(f'"{patient_index}" == patient_index')
            unique_slices = patient_slices['slice_index'].unique()
            for slice_index in unique_slices:
                index_dict['patient_index'].append(patient_index)
                index_dict['slice_index'].append(slice_index)
        self.index_dataframe = pd.DataFrame.from_dict(index_dict)

    def get_all_patient_slices(self, patient_index:str) -> List[int]:
        return self.dataframe_subset[self.dataframe_subset['patient_index'] == patient_index]['slice_index'].unique() # type:ignore

    def compute_slices_dataframe(self):
        print('Computing all slices dataframe...')
        slices_dict = {
            'patient_index' : [],
            'slice_index': []
        }
        for patient_index in range(1, 1012):
            patient_index = f'LIDC-IDRI-{format_index(patient_index)}'
            path_to_patient = self.path_to_processed_dataset.joinpath(patient_index)
            n_slices = len(list(path_to_patient.glob(f'slice_*')))
            for slice_index in range(n_slices):
                slices_dict['patient_index'].append(patient_index)
                slices_dict['slice_index'].append(slice_index)
        dataframe = pd.DataFrame.from_dict(slices_dict)
        dataframe.to_csv(self.slices_dataframe_path, index=False)

    def get_patient_dataset(self, patient_index:str) -> PatientDataset:
        '''
        This function is used mainly for evaluation.
        It takes a patient index and returns a Dataset object, that only contains the slices (and potentially annotations)
        of interest for a given patient. It could inherit functions compute_mask/reconstruction_tensor, but that is for another day.
        '''
        if self.pipeline == 'reconstruction':
            patient_annotation_dataframe = None
        else:
            patient_annotation_dataframe = self.current_dataframe[self.current_dataframe['patient_index'] == patient_index]
        return PatientDataset(
            self.path_to_processed_dataset,
            self.pipeline,
            patient_index,
            self.get_all_patient_slices(patient_index),
            annotations_dataframe=patient_annotation_dataframe,
            transform=self.transform
        )

    def compute_mask_tensor(self, patient_index: str, slice_index: int) -> torch.Tensor:
        mask = torch.zeros((512, 512), dtype=torch.bool)
        df_query = f'"{patient_index}" == patient_index & {slice_index} == slice_index'
        subset = self.current_dataframe.query(df_query)
        slice_nodules = subset['nodule_index'].unique()
        for nodule in slice_nodules:
            annotations:pd.Series = subset[subset['nodule_index'] == nodule]
            annotation_row = annotations.sample()
            nodule_index = int(annotation_row['nodule_index'].values[0])
            annotation_index = int(annotation_row['annotation_index'].values[0])
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
        if self.pipeline in ['joint', 'segmentation']:
            ### Compute mask tensor
            mask_tensor  = self.compute_mask_tensor(patient_index, slice_index)
            if self.transform is not None:
                reconstruction_tensor = self.transform["reconstruction_transforms"](
                    reconstruction_tensor
                )
                mask_tensor = self.transform["mask_transforms"](mask_tensor)
            return reconstruction_tensor, mask_tensor
        elif self.pipeline == 'reconstruction':
            if self.transform is not None:
                reconstruction_tensor = self.transform["reconstruction_transforms"](
                    reconstruction_tensor
                )
            return reconstruction_tensor

