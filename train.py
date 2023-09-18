import argparse
import pathlib
from datetime import datetime
from typing import Dict

import json
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # type:ignore
import torch
import pandas as pd

from metrics import PSNR  # type:ignore
from datasets import LIDC_IDRI
from backends.odl import ODLBackend
from models import load_network, Unet, LearnedPrimalDual
from utils import PyPlotImageWriter
from transforms import Normalise, ToFloat  # type:ignore

def loss_name_to_loss_function(loss_function_name: str):
    '''
    Parses a string to a pytorch loss function
    '''
    if loss_function_name == "MSE":
        return torch.nn.MSELoss()
    elif loss_function_name == "L1":
        return torch.nn.L1Loss()
    elif loss_function_name == "BCE":
        return torch.nn.BCELoss()
    else:
        raise NotImplementedError(
            f'Loss function called {loss_function_name} is not implemented, currently only ["MSE", "L1", "BCE"] are supported'
        )

def unpack_architecture_dicts(architecture_dict:Dict, device:torch.device, odl_backend=None) -> Dict[str, torch.nn.Module]:
    '''
    Parses a dictionnary containing the information about a network to the network object (torch.nn.Module) \\
    Raises error if the architecture name found is not implemented
    Returns a dictionnary of modules
    '''
    networks = {}
    for architecture_name, network_dict in architecture_dict.items():
        if architecture_name == 'reconstruction' or architecture_name == 'segmentation':
            network_name = network_dict['name']
            print(f'Unpacking {architecture_name} architecture: {network_name} network on device {device}')
            if architecture_name == 'reconstruction':
                if network_name =='lpd':
                    network = LearnedPrimalDual(odl_backend, network_dict, device) # type:ignore
                elif network_name =='filtered_backprojection':
                    network = None

                else:
                    raise NotImplementedError(f"{network_name} not implemented")

            elif architecture_name =='segmentation':

                if network_name =='Unet':
                    network = Unet(network_dict['unet_dict']).to(device)

                else:
                    raise NotImplementedError(f"{network_name} not implemented")

            else:
                raise NotImplementedError(f"{architecture_name} not implemented")

            networks[architecture_name] = network

    return networks

def load_networks(
    load_file_path:pathlib.Path,
    networks:Dict[str,torch.nn.Module],
    device:torch.device)->Dict[str,torch.nn.Module]:
    ### We check if the load file exists
    if load_file_path.is_file():
        print(f'Loading file at {load_file_path}')
        load_file = torch.load(load_file_path, map_location = device)
        for network_name, network in networks.items():
            if network_name in load_file:
                print(f'Loading {network_name} from load file')
                network.load_state_dict(load_file[network_name])
            else:
                print(f'No {network_name} key found in the load file, returning {network_name} network without initialisation')

    else:
        print(f'No saved file found at {load_file_path}, returning networks with no initialisation')
    return networks

def load_checkpoint(
    networks: Dict[str,torch.nn.Module],
    optimiser:torch.optim.Optimizer,
    checkpoint_load_path:pathlib.Path,
    map_location:torch.device
    ):
    '''
    Loads information from the checkpoint in the event of a stopped training (useful for HPC).
    Returns \\
    -> the networks with the state dict from the checkpoint
    \t if there is no checkpoint, returns the networks
    -> the optimiser with the state dict from the checkpoint
    \t if there is no checpoint, returns the optimiser
    -> the current row of the training plan
    \t if there is no checkpoint, returns 0
    -> the number of epochs already trained on
    \t if there is no checkpoint, returns 0
    '''
    if checkpoint_load_path.is_file():
        checkpoint = torch.load(checkpoint_load_path, map_location)
        ### We now iterate over the different things we have saved
        # 1) The networks
        for network_name, network in networks.items():
            network.load_state_dict(checkpoint[network_name])
        # 2) The optimiser
        optimiser.load_state_dict(checkpoint['optimiser'])
        # 3) The row of the training plan
        row = checkpoint['row']
        # 4) The current epoch of this checkpoint
        epoch = checkpoint['epoch']
    else:
        print(
            f"No file found at {checkpoint_load_path}, no initialisation"
        )  # type:ignore
        row = 0
        epoch = 0
    return networks, optimiser, row, epoch

def train_reconstruction(
    networks:Dict[str, torch.nn.Module],
    device:torch.device,
    training_plan:pd.DataFrame,
    checkpoint_file_path:pathlib.Path,
    save_file_path:pathlib.Path,
    verbose :bool):

    ## Loss functions
    reconstruction_loss_function = loss_name_to_loss_function(training_dict["reconstruction_loss"])
    psnr_loss = PSNR()

    ## Optimisers
    optimiser = torch.optim.Adam(
        lr = training_plan.iloc[0]['learning_rate'],
        params=networks['reconstruction'].parameters()
        )

    ## Load checkpoint
    networks, optimiser, checkpoint_row_index, checkpoint_epoch = load_checkpoint(
        networks,
        optimiser,
        checkpoint_file_path,
        device
    )

    ## Training loop
    for index, row in training_plan.iloc[checkpoint_row_index:].iterrows():
        ### Compute the number of trained epochs
        trained_epochs = training_plan.iloc[:checkpoint_row_index]['n_epochs'].sum()
        ### Compute the number of remaining epochs and the adequate learning rate
        n_planned_epochs = row['n_epochs']
        if index == checkpoint_row_index:
            ### if we are at the
            n_epochs = n_planned_epochs-checkpoint_epoch
            trained_epochs += checkpoint_epoch
        else:
            n_epochs = n_planned_epochs
        ### integer conversion of the data loaded from the checpoint file, or from the training plan
        n_epochs = int(n_epochs)
        trained_epochs = int(trained_epochs)

        ### Define the training dataset and dataloader
        ## Dataset
        lidc_idri_dataset = LIDC_IDRI(
            path_to_processed_dataset=DATASET_PATH,
            training_proportion = data_feeding_dict['training_proportion'],
            training = True,
            pipeline=pipeline,
            transform = transforms
        )
        ## Dataloader
        dataloader = DataLoader(
            dataset = lidc_idri_dataset,
            batch_size = data_feeding_dict['batch_size'],
            shuffle = data_feeding_dict['shuffle'],
            num_workers= data_feeding_dict['num_workers']
        )

        ### Set the appropriate learning rate value
        optimiser.param_groups[0]['lr'] = row['learning_rate']

        ### Loop over the epochs
        for epoch in range(n_epochs):
            print(f"Training epoch {epoch} / {n_epochs}...")
            for index, reconstruction in enumerate(dataloader):
                reconstruction = reconstruction.to(device)
                ## Re-sample
                sinogram = odl_backend.get_sinogram(reconstruction)
                sinogram = sinogram_transforms(sinogram)  # type:ignore

                ## Reconstruct
                optimiser.zero_grad()
                approximated_reconstruction, approximated_sinogram = networks['reconstruction'](sinogram)
                reconstruction_loss_value = reconstruction_loss_function(approximated_reconstruction, reconstruction)
                reconstruction_loss_value.backward()
                optimiser.step()

                if index % 50 == 0:
                    if verbose:
                        print(index)
                        print(f"Primal PSNR: {psnr_loss(approximated_reconstruction, reconstruction).item()}")
                        print(f"Primal MSE: {reconstruction_loss_value.item()}")

                    run_writer.add_scalar(
                        f"Primal PSNR",
                        psnr_loss(approximated_reconstruction, reconstruction).item(),
                        global_step=index + trained_epochs * dataloader.__len__()
                        )
                    run_writer.add_scalar(
                        f"Primal MSE",
                        reconstruction_loss_value.item(),
                        global_step=index + trained_epochs * dataloader.__len__()
                        )

                    tgt_rec = torch.cat(
                            (
                                display_transform(reconstruction[0, 0]),
                                display_transform(approximated_reconstruction[0, 0]),
                            ),  dim=1)

                    image_writer.write_image_tensor(tgt_rec,"target_reconstruction.jpg")

            trained_epochs += 1
            ### We checkpoint after each epoch
            torch.save({
                'reconstruction': networks['reconstruction'].state_dict(),
                'optimiser':optimiser.state_dict(),
                'row':row,
                'epoch':epoch
                }, checkpoint_file_path)
    torch.save({
        'reconstruction': networks['reconstruction'].state_dict(),
    }, save_file_path)

def train_segmentation(
    networks:Dict[str, torch.nn.Module],
    device:torch.device,
    training_plan:pd.DataFrame,
    checkpoint_file_path:pathlib.Path,
    save_file_path:pathlib.Path,
    verbose:bool):

    ## Loss functions
    segmentation_loss_function = loss_name_to_loss_function(training_dict["segmentation_loss"])

    ## Optimisers
    optimiser = torch.optim.Adam(
        lr = training_plan.iloc[0]['learning_rate'],
        params=networks['segmentation'].parameters()
        )

    ## Load checkpoint
    networks, optimiser, checkpoint_row_index, checkpoint_epoch = load_checkpoint(
        networks,
        optimiser,
        checkpoint_file_path,
        device
    )

    ## Training loop
    for index, row in training_plan.iloc[checkpoint_row_index:].iterrows():
        ### Compute the number of trained epochs
        trained_epochs = training_plan.iloc[:checkpoint_row_index]['n_epochs'].sum()
        ### Compute the number of remaining epochs and the adequate learning rate
        n_planned_epochs = row['n_epochs']
        if index == checkpoint_row_index:
            ### if we are at the
            n_epochs = n_planned_epochs-checkpoint_epoch
            trained_epochs += checkpoint_epoch
        else:
            n_epochs = n_planned_epochs
        ### integer conversion of the data loaded from the checpoint file, or from the training plan
        n_epochs = int(n_epochs)
        trained_epochs = int(trained_epochs)

        ### Define the training dataset and dataloader
        ## Get the query string
        query_string = f'{row["nodule_size"]} < annotation_size'
        print(f'Querying the dataset for {query_string} ')
        ## Dataset
        lidc_idri_dataset = LIDC_IDRI(
            path_to_processed_dataset=DATASET_PATH,
            training_proportion = data_feeding_dict['training_proportion'],
            training = True,
            pipeline=pipeline,
            query_string=query_string,
            transform = transforms
        )
        ## Dataloader
        dataloader = DataLoader(
            dataset = lidc_idri_dataset,
            batch_size = data_feeding_dict['batch_size'],
            shuffle = data_feeding_dict['shuffle'],
            num_workers= data_feeding_dict['num_workers']
        )
        ### Set the appropriate learning rate value
        optimiser.param_groups[0]['lr'] = row['learning_rate']

        for epoch in range(n_epochs):
            print(f"Training epoch {epoch} / {n_epochs}...")
            for index, data in enumerate(dataloader):
                reconstruction = data[0].to(device)
                mask = data[-1].to(device)

                ## Re-sample
                if data_feeding_dict['reconstructed']:
                    sinogram = odl_backend.get_sinogram(reconstruction)
                    sinogram = sinogram_transforms(sinogram)  # type:ignore
                    reconstruction, approximated_sinogram = networks['reconstruction'](sinogram)

                optimiser.zero_grad()
                ## Segment
                approximated_segmentation:torch.Tensor = networks['segmentation'](reconstruction)
                loss_segmentation = segmentation_loss_function(approximated_segmentation, mask)
                ## Get total loss
                loss_segmentation.backward()
                optimiser.step()

                if index % 50 == 0:
                    if verbose:
                        print(index)
                        print(f"Image BCE Loss: {loss_segmentation.item()}")
                    run_writer.add_scalar(
                        f"Image BCE Loss",
                        loss_segmentation.item(),
                        global_step=index + epoch * dataloader.__len__()
                    )
                    rec_seg_tgt = torch.cat(
                            (
                                display_transform(reconstruction[0, 0]),
                                display_transform(approximated_segmentation[0, 0]),
                                display_transform(mask[0, 0]),
                            ),  dim=1)

                    image_writer.write_image_tensor(rec_seg_tgt,"input_segmentation_tgt.jpg")

            if data_feeding_dict['reconstructed']:
                torch.save({
                    'reconstruction': networks['reconstruction'].state_dict(),
                    'segmentation': networks['segmentation'].state_dict(),
                    'optimiser':optimiser.state_dict(),
                    'row':row,
                    'epoch':epoch
                    }, checkpoint_file_path)
            else:
                torch.save({
                    'segmentation': networks['segmentation'].state_dict(),
                    'optimiser':optimiser.state_dict(),
                    'row':row,
                    'epoch':epoch
                    }, checkpoint_file_path)
        if data_feeding_dict['reconstructed']:
            torch.save({
                'reconstruction': networks['reconstruction'].state_dict(),
                'segmentation': networks['segmentation'].state_dict(),
            }, save_file_path)
        else:
            torch.save({
                'segmentation': networks['segmentation'].state_dict(),
            }, save_file_path)

def train_joint(
    networks:Dict[str, torch.nn.Module],
    device:torch.device,
    training_plan:pd.DataFrame,
    checkpoint_file_path:pathlib.Path,
    save_file_path:pathlib.Path,
    verbose:bool):

    ## Loss functions
    segmentation_loss_function = loss_name_to_loss_function(training_dict["segmentation_loss"])
    reconstruction_loss_function = loss_name_to_loss_function(training_dict["reconstruction_loss"])
    psnr_loss = PSNR()

    ## Optimisers
    optimiser = torch.optim.Adam(
        lr = training_plan.iloc[0]['learning_rate'],
        params=list(networks['reconstruction'].parameters()) + \
               list(networks['segmentation'].parameters())
        )

    ## Load checkpoint
    networks, optimiser, checkpoint_row_index, checkpoint_epoch = load_checkpoint(
        networks,
        optimiser,
        checkpoint_file_path,
        device
    )

    ## Training loop
    for index, row in training_plan.iloc[checkpoint_row_index:].iterrows():
        ### Compute the number of trained epochs
        trained_epochs = training_plan.iloc[:checkpoint_row_index]['n_epochs'].sum()
        ### Compute the number of remaining epochs and the adequate learning rate
        n_planned_epochs = row['n_epochs']
        if index == checkpoint_row_index:
            ### if we are at the
            n_epochs = n_planned_epochs-checkpoint_epoch
            trained_epochs += checkpoint_epoch
        else:
            n_epochs = n_planned_epochs
        ### integer conversion of the data loaded from the checpoint file, or from the training plan
        n_epochs = int(n_epochs)
        trained_epochs = int(trained_epochs)

        ### Define the training dataset and dataloader
        ## Get the query string
        query_string = f'{row["nodule_size"]} < annotation_size'
        print(f'Querying the dataset for {query_string} ')
        ## Dataset
        lidc_idri_dataset = LIDC_IDRI(
            path_to_processed_dataset=DATASET_PATH,
            training_proportion = data_feeding_dict['training_proportion'],
            training = True,
            pipeline=pipeline,
            query_string=query_string,
            transform = transforms
        )
        ## Dataloader
        dataloader = DataLoader(
            dataset = lidc_idri_dataset,
            batch_size = data_feeding_dict['batch_size'],
            shuffle = data_feeding_dict['shuffle'],
            num_workers= data_feeding_dict['num_workers']
        )
        ### Set the appropriate learning rate value
        optimiser.param_groups[0]['lr'] = row['learning_rate']

        for epoch in range(n_epochs):
            print(f"Training epoch {epoch} / {n_epochs}...")
            for index, data in enumerate(dataloader):
                reconstruction = data[0].to(device)
                mask = data[-1].to(device)

                ## Re-sample
                sinogram = odl_backend.get_sinogram(reconstruction)
                sinogram = sinogram_transforms(sinogram)  # type:ignore
                optimiser.zero_grad()
                approximated_reconstruction, approximated_sinogram = networks['reconstruction'](sinogram)
                loss_recontruction = reconstruction_loss_function(approximated_reconstruction, reconstruction)

                ## Segment
                approximated_segmentation:torch.Tensor = networks['segmentation'](approximated_reconstruction)
                loss_segmentation = segmentation_loss_function(approximated_segmentation, mask)
                ## Get total loss
                total_loss = training_dict['segmentation_loss_weighting']*loss_segmentation + \
                    (1-training_dict['segmentation_loss_weighting'])*loss_recontruction
                total_loss.backward()
                optimiser.step()

                if index % 50 == 0:
                    if verbose:
                        print(index)
                        print(f"Image BCE Loss: {loss_segmentation.item()}")
                    run_writer.add_scalar(
                        f"Primal PSNR",
                        psnr_loss(approximated_reconstruction, reconstruction).item(),
                        global_step=index + epoch * dataloader.__len__()
                        )

                    run_writer.add_scalar(
                        f"Image BCE Loss",
                        loss_segmentation.item(),
                        global_step=index + epoch * dataloader.__len__()
                        )

                    run_writer.add_scalar(
                        f"Total Loss",
                        total_loss.item(),
                        global_step=index + epoch * dataloader.__len__()
                        )
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
                    'reconstruction': networks['reconstruction'].state_dict(),
                    'segmentation': networks['segmentation'].state_dict(),
                    'optimiser':optimiser.state_dict(),
                    'row':row,
                    'epoch':epoch
                    }, checkpoint_file_path)

        torch.save({
                'reconstruction': networks['reconstruction'].state_dict(),
                'segmentation': networks['segmentation'].state_dict(),
            }, save_file_path)

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

    ## Models
    models_path = pathlib.Path(MODELS_PATH)
    ## The load path is stored in the json
    load_file_path = models_path / architecture_dict['load_path']
    ## The save path is computed from the folder in which the json file is, and its name
    save_file_path = models_path / f'{pipeline}/{experiment_folder_name}/{run_name}.pth'
    save_file_path.parent.mkdir(parents=True, exist_ok=True)
    ## The checkpoint path is computed from the folder in which the json file is, and its name
    checkpoint_file_path = models_path / f'{pipeline}/{experiment_folder_name}/{run_name}_checkpoint.pth'

    ## Device
    device = torch.device(architecture_dict["device_name"])

    networks = unpack_architecture_dicts(
        architecture_dict,
        device,
        odl_backend
        )

    ## Define and load network
    networks = load_networks(
        load_file_path,
        networks,
        device
        )

    ## Training plan
    training_plan = pd.DataFrame.from_dict(training_dict['training_plan'])

    if pipeline == 'reconstruction':
        train_reconstruction(
            networks,
            device,
            training_plan,
            checkpoint_file_path,
            save_file_path,
            verbose = VERBOSE_DICT[args.platform]
            )

    elif pipeline == 'segmentation':
        train_segmentation(
            networks,
            device,
            training_plan,
            checkpoint_file_path,
            save_file_path,
            verbose = VERBOSE_DICT[args.platform]
            )

    elif pipeline == 'joint':
        train_joint(
            networks,
            device,
            training_plan,
            checkpoint_file_path,
            save_file_path,
            verbose = VERBOSE_DICT[args.platform]
            )

    else:
        raise NotImplementedError