import argparse
import pathlib

import json
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter #type:ignore

from datasets import LIDC_IDRI
from backends.odl import ODLBackend
from train_functions import train_fbp, train_end_to_end, train_joint, train_reconstruction_network, train_segmentation_network
from utils import check_metadata, PyPlotImageWriter
from transforms import Normalise, ToFloat # type:ignore

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', required=False, default='reconstruction', help='fourier_filter, reconstruction, segmentation, joint, sequential or end_to_end')
    parser.add_argument('--metadata_path', required=True)
    parser.add_argument('--experiment_name', required=True)
    args = parser.parse_args()

    ## Metadata
    metadata_dict = dict(json.load(open(args.metadata_path)))

    ## Sanity checks
    check_metadata(metadata_dict)

    ## Unpacking dicts
    training_dict = metadata_dict["training_dict"]
    scan_parameter_dict = metadata_dict["scan_parameter_dict"]
    architecture_dict = metadata_dict['architecture_dict']
    dimension = metadata_dict['dimension']

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

    testing_lidc_idri_dataset = LIDC_IDRI(
        metadata_dict['pipeline'],
        odl_backend,
        training_dict['training_proportion'],
        'testing',
        training_dict['is_subset'],
        transform = transforms,
        subset = training_dict['subset']
        )
    testing_dataloader = DataLoader(
        testing_lidc_idri_dataset,
        1,
        shuffle=False,
        drop_last=False,
        num_workers=training_dict["num_workers"])

    image_writer = PyPlotImageWriter(pathlib.Path(f'images/{args.experiment_name}'))

    pathlib.Path(f'/local/scratch/public/ev373/runs').mkdir(parents=True, exist_ok=True)

    run_writer = SummaryWriter(
        log_dir = f'/local/scratch/public/ev373/runs/{args.experiment_name}'
    )

    if args.type == 'joint':
        train_joint(
            dimension=dimension,
            odl_backend=odl_backend,
            architecture_dict = architecture_dict,
            training_dict = training_dict,
            train_dataloader = training_dataloader,
            test_dataloader = testing_dataloader,
            image_writer = image_writer
        )

    elif args.type == 'reconstruction':
        train_reconstruction_network(
            dimension=dimension,
            odl_backend=odl_backend,
            architecture_dict = architecture_dict,
            training_dict = training_dict,
            train_dataloader = training_dataloader,
            image_writer = image_writer,
            run_writer = run_writer
        )

    elif args.type == 'segmentation':
        train_segmentation_network(
            architecture_dict = architecture_dict,
            training_dict = training_dict,
            train_dataloader = training_dataloader,
            test_dataloader = testing_dataloader,
            image_writer = image_writer
        )

    elif args.type == 'end_to_end':
        train_end_to_end(
            dimension=dimension,
            odl_backend=odl_backend,
            architecture_dict = architecture_dict,
            training_dict = training_dict,
            train_dataloader = training_dataloader,
            test_dataloader = testing_dataloader,
            image_writer = image_writer
        )

    elif args.type == 'fourier_filter':
        train_fbp(
            dimension=dimension,
            odl_backend=odl_backend,
            architecture_dict = architecture_dict,
            training_dict = training_dict,
            train_dataloader = training_dataloader,
            test_dataloader = testing_dataloader,
            image_writer = image_writer
        )

    else:
        raise ValueError(f'Wrong type value, must be fourier_filter, joint, sequential or end_to_end, not {args.type}') #type:ignore
