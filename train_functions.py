from typing import Dict
import pathlib

from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter #type:ignore
from torchvision.transforms import Compose
import matplotlib.pyplot as plt

from models import FourierFilteringModule, LearnedPrimalDual, Unet, load_network #type:ignore
from backends.odl import ODLBackend
from transforms import Normalise, PoissonSinogramTransform #type:ignore
from metrics import PSNR #type:ignore
from utils import PyPlotImageWriter

def loss_name_to_loss_function(loss_function_name:str):
    if loss_function_name == 'MSE':
        return torch.nn.MSELoss()
    elif loss_function_name == 'L1':
        return torch.nn.L1Loss()
    elif loss_function_name == 'BCE':
        return torch.nn.BCELoss()
    else:
        raise NotImplementedError(f'Loss function called {loss_function_name} is not implemented, currently only ["MSE", "L1", "BCE"] are supported')


def train_reconstruction_network(
        odl_backend:ODLBackend,
        architecture_dict:Dict,
        training_dict:Dict,
        train_dataloader:DataLoader,
        image_writer:PyPlotImageWriter,
        run_writer:SummaryWriter,
        save_folder_path:pathlib.Path,
        verbose=True
        ):

    reconstruction_dict = architecture_dict['reconstruction']
    dimension = reconstruction_dict['dimension']

    reconstruction_device = torch.device(reconstruction_dict['device_name'])

    if reconstruction_dict['name'] == 'lpd':
            reconstruction_net = LearnedPrimalDual(
            dimension = dimension,
            odl_backend = odl_backend,
            n_primal=reconstruction_dict['n_primal'],
            n_dual=reconstruction_dict['n_dual'],
            n_iterations = reconstruction_dict['lpd_n_iterations'],
            n_filters_primal = reconstruction_dict['lpd_n_filters_primal'],
            n_filters_dual = reconstruction_dict['lpd_n_filters_dual'],
            fourier_filtering = reconstruction_dict['fourier_filtering'],
            device = reconstruction_device
        )
    else:
        raise NotImplementedError(f"{reconstruction_dict['name']} not implemented")

    reconstruction_net = load_network(save_folder_path, reconstruction_net, reconstruction_dict['load_path'])


    reconstruction_loss = loss_name_to_loss_function(training_dict['reconstruction_loss'])
    sinogram_loss = loss_name_to_loss_function(training_dict['sinogram_loss'])

    psnr_loss = PSNR()

    optimiser = torch.optim.Adam(
        params = reconstruction_net.parameters(),
        lr = training_dict['learning_rate'],
        betas=(0.9,0.99)
    )

    sinogram_transforms = Normalise()

    reconstruction_model_save_path = pathlib.Path(save_folder_path)
    reconstruction_model_save_path.mkdir(exist_ok=True, parents=True)
    reconstruction_model_file_save_path = reconstruction_model_save_path.joinpath(reconstruction_dict['save_path'])

    for epoch in range(training_dict['n_epochs']):
        print(f"Training epoch {epoch} / {training_dict['n_epochs']}...")
        for index, reconstruction in enumerate(train_dataloader):

            reconstruction = reconstruction.to(reconstruction_device)
            optimiser.zero_grad()
            sinogram = odl_backend.get_sinogram(reconstruction)

            if dimension == 1:
                sinogram = torch.squeeze(sinogram, dim=1)

            sinogram = sinogram_transforms(sinogram)

            approximated_reconstruction, approximated_sinogram = reconstruction_net(sinogram)

            loss_recontruction = reconstruction_loss(approximated_reconstruction, reconstruction)
            loss_sinogram = sinogram_loss(approximated_sinogram, sinogram)

            total_loss = (1-training_dict['dual_loss_weighting'])*loss_recontruction + training_dict['dual_loss_weighting']*loss_sinogram
            total_loss.backward()

            optimiser.step()

            if index %10 == 0:
                if verbose:
                    print(f'\n Metrics at step {index} of epoch {epoch}')
                    print(f'Image {training_dict["reconstruction_loss"]} : {loss_recontruction.item()}')
                    print(f'Image PSNR : {psnr_loss(approximated_reconstruction, reconstruction).item()}')
                    print(f'Sinogram {training_dict["sinogram_loss"]} : {loss_sinogram.item()}')
                    print(f'Sinogram PSNR : {psnr_loss(approximated_sinogram, sinogram).item()}')
                run_writer.add_scalar(f'Image {training_dict["reconstruction_loss"]} Loss', loss_recontruction.item(), global_step=index+epoch*train_dataloader.__len__())
                run_writer.add_scalar('Image PSNR Loss', psnr_loss(approximated_reconstruction, reconstruction).item(), global_step=index+epoch*train_dataloader.__len__())
                run_writer.add_scalar(f'Sinogram {training_dict["sinogram_loss"]} Loss', loss_sinogram.item(), global_step=index+epoch*train_dataloader.__len__())
                run_writer.add_scalar('Sinogram PSNR Loss', psnr_loss(approximated_sinogram, sinogram).item(), global_step=index+epoch*train_dataloader.__len__())
                image_writer.write_image_tensor(approximated_reconstruction, 'current_reconstruction.jpg')
                image_writer.write_image_tensor(reconstruction, 'reconstruction_target.jpg')
                image_writer.write_image_tensor(approximated_sinogram.unsqueeze(1), 'current_sinogram.jpg')
                image_writer.write_image_tensor(sinogram.unsqueeze(1), 'sinogram_target.jpg')

        torch.save(reconstruction_net.state_dict(), reconstruction_model_file_save_path)

def train_segmentation_network(
        odl_backend:ODLBackend,
        architecture_dict:Dict,
        training_dict:Dict,
        train_dataloader:DataLoader,
        image_writer:PyPlotImageWriter,
        run_writer:SummaryWriter,
        save_folder_path:pathlib.Path,
        verbose=True
        ):

    segmentation_device = torch.device(architecture_dict['segmentation']['device_name'])

    if architecture_dict['segmentation']['name'] == 'Unet':
        segmentation_net = Unet(
            dimension=2,
            n_channels_input  = architecture_dict['segmentation']['Unet_input_channels'],
            n_channels_output = architecture_dict['segmentation']['Unet_output_channels'],
            n_filters = architecture_dict['segmentation']['Unet_n_filters'],
            regression=False
            ).to(segmentation_device)
    else:
        raise NotImplementedError(f"{architecture_dict['segmentation']['name']} not implemented")

    segmentation_net = load_network(save_folder_path, segmentation_net, architecture_dict['segmentation']['load_path'])

    segmentation_loss = loss_name_to_loss_function(training_dict['segmentation_loss'])

    optimiser = torch.optim.Adam(
        params = segmentation_net.parameters(),
        lr = training_dict['learning_rate'],
        betas=(0.9,0.99)
        )

    if training_dict['reconstructed']:
        ## Unpacking dict
        reconstruction_dict = architecture_dict['reconstruction']
        dimension = reconstruction_dict['dimension']
        ## Define reconstruction device
        reconstruction_device = torch.device(reconstruction_dict['device_name'])

        ## Load reconstruction Network
        if reconstruction_dict['name'] == 'lpd':
            reconstruction_net = LearnedPrimalDual(
            dimension = dimension,
            odl_backend = odl_backend,
            n_primal=reconstruction_dict['n_primal'],
            n_dual=reconstruction_dict['n_dual'],
            n_iterations = reconstruction_dict['lpd_n_iterations'],
            n_filters_primal = reconstruction_dict['lpd_n_filters_primal'],
            n_filters_dual = reconstruction_dict['lpd_n_filters_dual'],
            fourier_filtering = reconstruction_dict['fourier_filtering'],
            device = reconstruction_device
        )

        elif reconstruction_dict['name'] == 'fourier_filtering':
            reconstruction_net = FourierFilteringModule(
                dimension=dimension,
                n_measurements=odl_backend.angle_partition_dict['shape'],
                detector_size=odl_backend.detector_partition_dict['shape'],
                device = segmentation_device,
                filter_name=reconstruction_dict['filter_name'],
                training_mode=reconstruction_dict['train']
            )
        else:
            raise NotImplementedError(f"{reconstruction_dict['name']} not implemented")

        try:
            reconstruction_net = load_network(save_folder_path, reconstruction_net, reconstruction_dict['load_path'])
        except KeyError:
                print('No save_path found, loading default model')

        reconstruction_net.eval()
        ## Define sinogram transform
        sinogram_transforms = Normalise()

    segmentation_model_save_path = pathlib.Path(save_folder_path)
    segmentation_model_save_path.mkdir(exist_ok=True, parents=True)
    segmentation_model_file_save_path = segmentation_model_save_path.joinpath(architecture_dict['segmentation']['save_path'])

    for epoch in range(training_dict['n_epochs']):
        print(f"Training epoch {epoch} / {training_dict['n_epochs']}...")
        for index, data in enumerate(train_dataloader):
            reconstruction = data[0].to(segmentation_device)
            mask = data[-1].to(segmentation_device)
            if training_dict['reconstructed']:
                with torch.no_grad():
                    ## Re-sample
                    sinogram = odl_backend.get_sinogram(reconstruction)
                    if dimension == 1: #type:ignore
                        sinogram = torch.squeeze(sinogram, dim=1)
                    sinogram = sinogram_transforms(sinogram) #type:ignore
                    ## Reconstruct
                    if reconstruction_dict['name'] == 'lpd': #type:ignore
                        reconstruction = reconstruction_net(sinogram, just_infer=True) #type:ignore
                    elif reconstruction_dict['name'] == 'fourier_filtering': #type:ignore
                        filtered_sinogram:torch.Tensor = reconstruction_net(sinogram) #type:ignore
                        reconstruction = odl_backend.get_reconstruction(filtered_sinogram.unsqueeze(1))
                    else:
                        raise NotImplementedError
            optimiser.zero_grad()
            approximated_segmentation = segmentation_net(reconstruction)
            loss_segmentation  = segmentation_loss(approximated_segmentation, mask)

            loss_segmentation.backward()

            optimiser.step()

            print(f'Segmentation Loss : {loss_segmentation.item():.5f}')

            if index %10 == 0:
                if verbose:
                    print(f'\n Metrics at step {index} of epoch {epoch}')
                    print(f'Image BCE Loss : {loss_segmentation.item()}')
                run_writer.add_scalar(f'Image BCE Loss', loss_segmentation.item(), global_step=index+epoch*train_dataloader.__len__())
                image_writer.write_image_tensor(reconstruction, 'current_reconstruction.jpg')
                image_writer.write_image_tensor(approximated_segmentation, 'current_segmentation.jpg')
                image_writer.write_image_tensor(mask, 'target_segmentation.jpg')

        torch.save(segmentation_net.state_dict(), segmentation_model_file_save_path)

