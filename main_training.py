import argparse

import json
from torchvision.transforms import Compose
from torch.utils.data import DataLoader

from datasets import LIDC_IDRI
from backends.odl import ODLBackend
from train_functions import train_end_to_end, train_joint, train_reconstruction_network, train_segmentation_network
from utils import check_metadata
from transforms import Normalise # type:ignore

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', required=False, default='reconstruction', help='reconstruction, segmentation, joint, sequential or end_to_end')
    parser.add_argument('--metadata_path', required=False, default='metadata_folder/reconstruction_01_06_23.json')
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
        "reconstruction_transforms":Compose(Normalise()),
        "mask_transforms":None
    }

    ## Dataset and Dataloader
    training_lidc_idri_dataset = LIDC_IDRI(
        metadata_dict['pipeline'],
        training_dict['training_proportion'],
        'training',
        transform = transforms
        )
    training_dataloader = DataLoader(
        training_lidc_idri_dataset,
        training_dict["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=training_dict["num_workers"])

    testing_lidc_idri_dataset = LIDC_IDRI(
        metadata_dict['pipeline'],
        training_dict['training_proportion'],
        'testing',
        transform = transforms
        )
    testing_dataloader = DataLoader(
        testing_lidc_idri_dataset,
        training_dict["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=training_dict["num_workers"])

    if args.type == 'joint':
        train_joint(
            dimension=dimension,
            odl_backend=odl_backend,
            architecture_dict = architecture_dict,
            training_dict = training_dict,
            train_dataloader = training_dataloader,
            test_dataloader = testing_dataloader
        )

    elif args.type == 'reconstruction':
        train_reconstruction_network(
            dimension=dimension,
            odl_backend=odl_backend,
            architecture_dict = architecture_dict,
            training_dict = training_dict,
            train_dataloader = training_dataloader,
            test_dataloader = testing_dataloader
        )

    elif args.type == 'segmentation':
        train_segmentation_network(
            dimension=dimension,
            architecture_dict = architecture_dict,
            training_dict = training_dict,
            train_dataloader = training_dataloader,
            test_dataloader = testing_dataloader
        )

    elif args.type == 'end_to_end':
        train_end_to_end(
            dimension=dimension,
            odl_backend=odl_backend,
            architecture_dict = architecture_dict,
            training_dict = training_dict,
            train_dataloader = training_dataloader,
            test_dataloader = testing_dataloader
        )

    else:
        raise ValueError(f'Wrong type value, must be joint, sequential or end_to_end, not {parser.type}') #type:ignore
