import argparse
import pathlib
from datetime import datetime
from typing import Dict

import json
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # type:ignore
import torch

from metrics import PSNR  # type:ignore
from datasets import LIDC_IDRI_SEGMENTATIONS
from backends.odl import ODLBackend
from train_functions import load_network, loss_name_to_loss_function, unpack_architecture_dicts
from utils import PyPlotImageWriter
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

    ## Writers
    image_writer = PyPlotImageWriter(
        pathlib.Path(f"images") / pipeline / experiment_folder_name / run_name
    )

    run_writer = SummaryWriter(
        log_dir=pathlib.Path(RUNS_PATH) / pipeline / experiment_folder_name / run_name
    )

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
    display_transform = Normalise()
    sinogram_transforms = Normalise()
    ## Training plan
    training_plan = {
        1024:200,
        768:200,
        512:200,
        256:200
    }
    base_lr = 1e-5
    lr_scheduler ={
        1024: base_lr,
        768: base_lr/2,
        512:base_lr/4,
        256:base_lr/8
    }

    ## Models
    models_path = pathlib.Path(MODELS_PATH)
    save_file_path = models_path / f'{pipeline}/{experiment_folder_name}/{run_name}.pth'
    save_file_path.parent.mkdir(parents=True, exist_ok=True)

    networks = unpack_architecture_dicts(
        architecture_dict,
        odl_backend
        )

    segmentation_dict = architecture_dict['segmentation']
    segmentation_network = networks['segmentation']
    segmentation_network = load_network(
        models_path,
        segmentation_network,
        segmentation_dict['load_path'],
        map_location=segmentation_dict['device_name']
        )
    segmentation_network.train()
    segmentation_device = torch.device(segmentation_dict["device_name"])

    reconstruction_dict = architecture_dict['reconstruction']
    reconstruction_network = networks['reconstruction']
    reconstruction_network = load_network(
        models_path,
        reconstruction_network,
        reconstruction_dict['load_path'],
        map_location=reconstruction_dict['device_name']
        )
    reconstruction_device = torch.device(reconstruction_dict["device_name"])

    if pipeline == 'joint':
        reconstruction_network.train()

    elif pipeline == 'sequential':
        reconstruction_network.eval()
        reconstruction_network.requires_grad_(False)

    else:
        raise ValueError

    segmentation_loss = loss_name_to_loss_function(training_dict["segmentation_loss"])
    reconstruction_loss = loss_name_to_loss_function(training_dict["reconstruction_loss"])

    old_index = 0

    for nodule_size, n_epochs in training_plan.items():
        print(f'Processing nodule with size superior to {nodule_size}')
        print(f'Learning rate: {lr_scheduler[nodule_size]}')
        ## Craft query
        query_string = f'{nodule_size} < nodule_size'

        ## Dataset
        dataset = LIDC_IDRI_SEGMENTATIONS(
            path_to_processed_dataset=DATASET_PATH,
            training_proportion = data_feeding_dict['training_proportion'],
            training = True,
            query_string = query_string,
            transform = transforms
        )

        print(f'There are {dataset.__len__()} slices in the dataset')
        ## Dataloader
        dataloader = DataLoader(
        dataset,
        data_feeding_dict["batch_size"],
        shuffle=data_feeding_dict['shuffle'],
        drop_last=True,
        num_workers=data_feeding_dict["num_workers"],
        )

        if pipeline == 'sequential':
            optimiser = torch.optim.Adam(
            params=segmentation_network.parameters(),
            lr=lr_scheduler[nodule_size],
            betas=(0.9, 0.99),
            )
        elif pipeline == 'joint':
            optimiser = torch.optim.Adam(
            params= list(reconstruction_network.parameters()) + \
                    list(segmentation_network.parameters()),
            lr=lr_scheduler[nodule_size],
            betas=(0.9, 0.99),
            )
        else:
            raise ValueError

        psnr_loss = PSNR()

        for epoch in range(n_epochs):
            print(f"Training epoch {epoch} / {n_epochs}...")
            for index, data in enumerate(dataloader):
                reconstruction = data[0].to(segmentation_device)
                mask = data[-1].to(segmentation_device)

                ## Re-sample
                sinogram = odl_backend.get_sinogram(reconstruction)
                sinogram = sinogram_transforms(sinogram)  # type:ignore

                optimiser.zero_grad()
                ## Reconstruct
                approximated_reconstruction, approximated_sinogram = reconstruction_network(sinogram)
                loss_recontruction = reconstruction_loss(approximated_reconstruction, reconstruction)
                ## Segment
                approximated_segmentation:torch.Tensor = segmentation_network(approximated_reconstruction)
                loss_segmentation = segmentation_loss(approximated_segmentation, mask)
                ## Get total loss
                total_loss = training_dict['segmentation_loss_weighting']*loss_segmentation + \
                    (1-training_dict['segmentation_loss_weighting'])*loss_recontruction

                optimiser.step()

                if index % 50 == 0:
                    run_writer.add_scalar(f"Primal PSNR", psnr_loss(approximated_reconstruction, reconstruction).item(),global_step=index + epoch * dataloader.__len__())
                    run_writer.add_scalar(f"Image BCE Loss", loss_segmentation.item(),global_step=index + epoch * dataloader.__len__())
                    run_writer.add_scalar(f"Total Loss", total_loss.item(),global_step=index + epoch * dataloader.__len__())

                    targets = torch.cat(
                            (
                                display_transform(reconstruction[0, 0]),
                                display_transform(mask[0, 0]),
                            ),  dim=1)

                    approxs = torch.cat(
                            (
                                display_transform(approximated_reconstruction[0, 0]),
                                display_transform(approximated_segmentation[0, 0]),
                            ),  dim=1)

                    image_writer.write_image_tensor(torch.cat((targets,approxs),dim=0,),"input_segmentation_tgt.jpg")

            torch.save({
                'reconstruction_net': reconstruction_network.state_dict(),
                'segmentation_net': segmentation_network.state_dict(),
                'optimiser':optimiser.state_dict(),
                }, save_file_path)


        old_index += n_epochs * dataloader.__len__()