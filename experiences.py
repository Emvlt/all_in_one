import argparse
import pathlib

import json
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter #type:ignore

from datasets import LIDC_IDRI
from backends.odl import ODLBackend
from train_functions import train_reconstruction_network, train_segmentation_network
from utils import check_metadata, PyPlotImageWriter
from transforms import Normalise, ToFloat # type:ignore

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_path', required=True)
    parser.add_argument('--platform', required=True)
    args = parser.parse_args()

    print(f'Running code on {args.platform}')
    ## Unpacking paths
    paths_dict = dict(json.load(open('paths_dict.json')))[args.platform]
    MODELS_PATH = pathlib.Path(paths_dict['MODELS_PATH'])
    RUNS_PATH = pathlib.Path(paths_dict['RUNS_PATH'])
    DATASET_PATH = pathlib.Path(paths_dict['DATASET_PATH'])

    ## Unpacking metadata
    metadata_path = pathlib.Path(args.metadata_path)
    pipeline = metadata_path.parent.parent
    experiment_folder_name = metadata_path.parent
    run_name = metadata_path.stem

    metadata_dict = dict(json.load(open(metadata_path)))
    training_dict = metadata_dict["training_dict"]
    scan_parameter_dict = metadata_dict["scan_parameter_dict"]
    architecture_dict = metadata_dict['architecture_dict']
    dimension = metadata_dict['dimension']

    ## Sanity checks
    check_metadata(metadata_dict)

    ## Instanciate backend
    odl_backend = ODLBackend()
    odl_backend.initialise_odl_backend_from_metadata_dict(scan_parameter_dict)

    ## Transforms
    transforms = {
        "reconstruction_transforms":Compose([ToFloat(), Normalise()]),
        "mask_transforms":Compose([ToFloat()])
    }

    ## Dataset and Dataloader
    training_lidc_idri_dataset = LIDC_IDRI(
        DATASET_PATH,
        metadata_dict['pipeline'],
        odl_backend,
        training_dict['training_proportion'],
        'training',
        training_dict['is_subset'],
        transform = transforms,
        subset = training_dict['subset']
        )
    training_dataloader = DataLoader(
        training_lidc_idri_dataset,
        training_dict["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=training_dict["num_workers"])

    image_writer = PyPlotImageWriter(pathlib.Path(f'images/{pipeline}/{experiment_folder_name}/{run_name}'))

    run_writer = SummaryWriter(
        log_dir = pathlib.Path(RUNS_PATH).joinpath(f'{pipeline}/{experiment_folder_name}/{run_name}')
    )

    models_path = pathlib.Path(MODELS_PATH).joinpath(f'{pipeline}/{experiment_folder_name}')

    if pipeline == 'reconstruction':
        train_reconstruction_network(
            dimension=dimension,
            odl_backend=odl_backend,
            architecture_dict = architecture_dict,
            training_dict = training_dict,
            train_dataloader = training_dataloader,
            image_writer = image_writer,
            run_writer = run_writer,
            save_folder_path = models_path
        )
    elif pipeline == 'segmentation':
        train_segmentation_network(
            dimension=dimension,
            odl_backend=odl_backend,
            architecture_dict = architecture_dict,
            training_dict = training_dict,
            train_dataloader = training_dataloader,
            image_writer = image_writer,
            run_writer = run_writer,
            save_folder_path = models_path
        )

    else:
        raise ValueError(f'Wrong type value, must be fourier_filter, joint, sequential or end_to_end, not {args.type}') #type:ignore
