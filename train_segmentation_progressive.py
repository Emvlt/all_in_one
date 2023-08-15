import argparse
import pathlib
from datetime import datetime
from typing import Dict

import json
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # type:ignore
import matplotlib.pyplot as plt
import torch

from datasets import LIDC_IDRI, LIDC_IDRI_SEGMENTATIONS
from backends.odl import ODLBackend
from train_functions import load_network, loss_name_to_loss_function, unpack_architecture_dicts
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

    training_plan = {
        #1024:200,
        #768:200,
        #512:200,
        256:200
    }
    lr_scheduler ={
        1024: training_dict['learning_rate'],
        768: training_dict['learning_rate']/2,
        512:training_dict['learning_rate']/4,
        256:training_dict['learning_rate']/8
    }

    image_writer = PyPlotImageWriter(
        pathlib.Path(f"images") / pipeline / experiment_folder_name / run_name
    )

    run_writer = SummaryWriter(
        log_dir=pathlib.Path(RUNS_PATH) / pipeline / experiment_folder_name / run_name
    )
    ### Format hyperparameters for registration
    # hparams = unpack_hparams(metadata_dict)
    #run_writer.add_hparams(hparams, metric_dict = {})
    models_path = pathlib.Path(MODELS_PATH)
    save_file_path = models_path / f'{pipeline}/{experiment_folder_name}/{run_name}.pth'
    save_file_path.parent.mkdir(parents=True, exist_ok=True)

    segmentation_dict = architecture_dict["segmentation"]
    segmentation_device = torch.device(segmentation_dict["device_name"])

    networks = unpack_architecture_dicts(architecture_dict, odl_backend)
    segmentation_net = networks['segmentation']
    segmentation_net = load_network(models_path, segmentation_net, segmentation_dict["load_path"])

    segmentation_loss = loss_name_to_loss_function(training_dict["segmentation_loss"])

    display_transform = Normalise()

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

        optimiser = torch.optim.Adam(
        params=segmentation_net.parameters(),
        lr=lr_scheduler[nodule_size],
        betas=(0.9, 0.99),
        )

        for epoch in range(n_epochs):
            print(f"Training epoch {epoch} / {n_epochs}...")
            for index, data in enumerate(dataloader):
                reconstruction = data[0].to(segmentation_device)
                mask = data[-1].to(segmentation_device)

                optimiser.zero_grad()
                approximated_segmentation:torch.Tensor = segmentation_net(reconstruction)
                loss_segmentation = segmentation_loss(approximated_segmentation, mask)
                loss_segmentation.backward()
                optimiser.step()

                if index % 10 == 0:

                    print(f"\n Metrics at step {index} of epoch {epoch}")
                    print(f"Image BCE Loss : {loss_segmentation.item()}")
                    run_writer.add_scalar(
                        f"Image BCE Loss",
                        loss_segmentation.item(),
                        global_step=old_index + index + epoch * dataloader.__len__(),
                    )
                    image_writer.write_image_tensor(
                        torch.cat(
                            (
                                display_transform(reconstruction[0, 0]),
                                display_transform(approximated_segmentation[0, 0]),
                                display_transform(mask[0, 0]),
                            ),
                            dim=1,
                        ),
                        "input_segmentation_tgt.jpg",
                    )
            torch.save(segmentation_net.state_dict(), save_file_path)#

        old_index += n_epochs * dataloader.__len__()