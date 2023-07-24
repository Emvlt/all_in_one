import argparse
import pathlib
from datetime import datetime
from typing import Dict

import json
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # type:ignore

from datasets import LIDC_IDRI
from backends.odl import ODLBackend
from train_functions import train_reconstruction_network, train_segmentation_network
from utils import PyPlotImageWriter, unpack_hparams
from metadata_checker import check_metadata
from transforms import Normalise, ToFloat  # type:ignore

VERBOSE_DICT ={
    'holly-b':True,
    'hpc':False
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_path", required=True)
    parser.add_argument("--platform", required=False, default='holly-b')
    args = parser.parse_args()

    ## Unpacking paths
    paths_dict = dict(json.load(open("paths_dict.json")))[args.platform]
    MODELS_PATH = pathlib.Path(paths_dict["MODELS_PATH"])
    RUNS_PATH = pathlib.Path(paths_dict["RUNS_PATH"])
    DATASET_PATH = pathlib.Path(paths_dict["DATASET_PATH"])

    ## Unpacking metadata
    metadata_path = pathlib.Path(args.metadata_path)
    pipeline = metadata_path.parent.parent.stem
    experiment_folder_name = metadata_path.parent.stem
    run_name = metadata_path.stem
    print(
        f"Running {pipeline} pipeline for {experiment_folder_name} experiment folder: experience {run_name} running on {args.platform}"
    )

    metadata_dict = dict(json.load(open(metadata_path)))
    data_feeding_dict = metadata_dict["data_feeding_dict"]
    training_dict = metadata_dict["training_dict"]
    architecture_dict = metadata_dict["architecture_dict"]

    ## Sanity checks
    check_metadata(metadata_dict, file_path=metadata_path)

    ## Instanciate backend
    odl_backend = ODLBackend()
    try:
        scan_parameter_dict = metadata_dict["scan_parameter_dict"]
        odl_backend.initialise_odl_backend_from_metadata_dict(scan_parameter_dict)
    except KeyError:
        print("No scanning dict in metadata, passing...")

    ## Transforms
    transforms = {
        "reconstruction_transforms": Compose([ToFloat(), Normalise()]),
        "mask_transforms": Compose([ToFloat()]),
    }

    ## Dataset and Dataloader
    if data_feeding_dict["is_subset"]:
        training_lidc_idri_dataset = LIDC_IDRI(
            DATASET_PATH,
            str(pipeline),
            odl_backend,
            data_feeding_dict["training_proportion"],
            data_feeding_dict["train"],
            data_feeding_dict["is_subset"],
            transform=transforms,
            subset=data_feeding_dict['subset']
        )
    else:
        training_lidc_idri_dataset = LIDC_IDRI(
            DATASET_PATH,
            str(pipeline),
            odl_backend,
            data_feeding_dict["training_proportion"],
            data_feeding_dict["train"],
            data_feeding_dict["is_subset"],
            transform=transforms
        )

    print(len(training_lidc_idri_dataset.training_patients_list))
    d = training_lidc_idri_dataset.patient_id_to_slices_of_interest
    total_slices = 0
    for patient_name, patient_dict in d.items():
        n_slices = len(patient_dict.keys())
        total_slices += n_slices
        print(f'Patient {patient_name} has {n_slices} slices of interest: {list(patient_dict.keys())}')
    print(f'There is a total of {total_slices} in the dataset')