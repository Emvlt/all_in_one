from typing import Dict
import pathlib

from torch.utils.data import DataLoader
import torch
from torch.utils.tensorboard import SummaryWriter  # type:ignore

from models import FourierFilteringModule, LearnedPrimalDual, Unet, load_network  # type:ignore
from backends.odl import ODLBackend
from transforms import Normalise, PoissonSinogramTransform  # type:ignore
from metrics import PSNR  # type:ignore
from utils import PyPlotImageWriter


def loss_name_to_loss_function(loss_function_name: str):
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

def unpack_architecture_dicts(architecture_dict:Dict, odl_backend=None) -> Dict[str, torch.nn.Module]:
    networks = {}
    for architecture_name, network_dict in architecture_dict.items():
        network_name = network_dict['name']
        current_device = network_dict['device_name']
        print(f'Unpacking {architecture_name} architecture: {network_name} network on device {current_device}')
        if architecture_name == 'reconstruction':
            if network_name =='lpd':
                network = LearnedPrimalDual(odl_backend, network_dict, current_device) # type:ignore
            elif network_name == 'fourier_filtering_module':
                network = FourierFilteringModule(
                    network_dict,
                    n_measurements=odl_backend.angle_partition_dict["shape"], # type:ignore
                    detector_size=odl_backend.detector_partition_dict["shape"], # type:ignore
                    device=current_device
                    )

            else:
                raise NotImplementedError(f"{network_name} not implemented")

        elif architecture_name =='segmentation':

            if network_name =='Unet':
                network = Unet(network_dict).to(current_device)

            else:
                raise NotImplementedError(f"{network_name} not implemented")

        else:
            raise NotImplementedError(f"{architecture_name} not implemented")

        networks[architecture_name] = network

    return networks

def train_reconstruction_network(
    odl_backend: ODLBackend,
    architecture_dict: Dict,
    training_dict: Dict,
    train_dataloader: DataLoader,
    image_writer: PyPlotImageWriter,
    run_writer: SummaryWriter,
    load_folder_path:pathlib.Path,
    save_file_path: pathlib.Path,
    verbose=True,
):

    reconstruction_dict = architecture_dict["reconstruction"]

    reconstruction_device = torch.device(reconstruction_dict["device_name"])

    reconstruction_net = unpack_architecture_dicts(architecture_dict, odl_backend)['reconstruction']
    reconstruction_net = load_network(load_folder_path, reconstruction_net, reconstruction_dict["load_path"])

    print(f'Number of network parameters: { sum(p.numel() for p in reconstruction_net.parameters())}')

    reconstruction_loss = loss_name_to_loss_function(training_dict["reconstruction_loss"])
    sinogram_loss = loss_name_to_loss_function(training_dict["sinogram_loss"])

    psnr_loss = PSNR()

    optimiser = torch.optim.Adam(
        params=reconstruction_net.parameters(),
        lr=training_dict["learning_rate"],
        betas=(0.9, 0.99),
    )

    sinogram_transforms = Normalise()
    display_transforms = Normalise()

    for epoch in range(training_dict["n_epochs"]):
        print(f"Training epoch {epoch} / {training_dict['n_epochs']}...")
        for index, reconstruction in enumerate(train_dataloader):

            reconstruction = reconstruction.to(reconstruction_device)
            optimiser.zero_grad()
            sinogram = odl_backend.get_sinogram(reconstruction)

            sinogram = sinogram_transforms(sinogram)

            approximated_reconstruction, approximated_sinogram = reconstruction_net(sinogram)

            loss_recontruction = reconstruction_loss(approximated_reconstruction, reconstruction)
            loss_sinogram = sinogram_loss(approximated_sinogram, sinogram)

            total_loss = (1 - training_dict["dual_loss_weighting"]) * loss_recontruction + training_dict["dual_loss_weighting"] * loss_sinogram
            total_loss.backward()

            optimiser.step()

            if index % 10 == 0:
                if verbose:
                    print(f"\n Metrics at step {index} of epoch {epoch}")
                    print(f'Primal {training_dict["reconstruction_loss"]} : {loss_recontruction.item()}')
                    print(f"Primal PSNR : {psnr_loss(approximated_reconstruction, reconstruction).item()}")
                    print(f'Dual {training_dict["sinogram_loss"]} : {loss_sinogram.item()}')
                    print(f"Dual PSNR : {psnr_loss(approximated_sinogram, sinogram).item()}")

                run_writer.add_scalar(
                    f'Primal {training_dict["reconstruction_loss"]} Loss',
                    loss_recontruction.item(),
                    global_step=index + epoch * train_dataloader.__len__(),
                )
                run_writer.add_scalar(
                    "Primal PSNR Loss",
                    psnr_loss(approximated_reconstruction, reconstruction).item(),
                    global_step=index + epoch * train_dataloader.__len__(),
                )
                run_writer.add_scalar(
                    f'Dual {training_dict["sinogram_loss"]} Loss',
                    loss_sinogram.item(),
                    global_step=index + epoch * train_dataloader.__len__(),
                )
                run_writer.add_scalar(
                    "Dual PSNR Loss",
                    psnr_loss(approximated_sinogram, sinogram).item(),
                    global_step=index + epoch * train_dataloader.__len__(),
                )
                run_writer.add_scalar(
                    f'Primal {training_dict["reconstruction_loss"]} / Dual {training_dict["sinogram_loss"]}',
                    (loss_recontruction/loss_sinogram).item(),
                    global_step=index + epoch * train_dataloader.__len__(),
                )

                if odl_backend.angle_partition_dict['shape'] < odl_backend.detector_partition_dict['shape']:
                    display_dim = 0
                else:
                    display_dim = 1
                image_writer.write_image_tensor(
                    torch.cat(
                        (
                            display_transforms(sinogram),
                            display_transforms(approximated_sinogram),
                        ),
                        dim=display_dim+2,
                    ),
                    "current_sinogram_approximation_target.jpg",
                )

                image_writer.write_image_tensor(
                    torch.cat(
                        (
                            display_transforms(reconstruction[0, 0]),
                            display_transforms(approximated_reconstruction[0, 0]),
                        ),
                        dim=1,
                    ),
                    "current_image_approximation_target.jpg",
                )

        torch.save(reconstruction_net.state_dict(), save_file_path)

def train_segmentation_network(
    odl_backend: ODLBackend,
    architecture_dict: Dict,
    training_dict: Dict,
    train_dataloader: DataLoader,
    image_writer: PyPlotImageWriter,
    run_writer: SummaryWriter,
    load_folder_path:pathlib.Path,
    save_file_path: pathlib.Path,
    verbose=True,
):

    segmentation_device = torch.device(architecture_dict["segmentation"]["device_name"])

    networks = unpack_architecture_dicts(architecture_dict, odl_backend)
    segmentation_net = networks['segmentation']
    segmentation_net = load_network(load_folder_path, segmentation_net, architecture_dict["segmentation"]["load_path"])

    segmentation_loss = loss_name_to_loss_function(training_dict["segmentation_loss"])

    optimiser = torch.optim.Adam(
        params=segmentation_net.parameters(),
        lr=training_dict["learning_rate"],
        betas=(0.9, 0.99),
    )

    if training_dict["reconstructed"]:
        reconstruction_dict = architecture_dict["reconstruction"]
        ## Define reconstruction device
        reconstruction_net = networks['reconstruction']
        try:
            reconstruction_net = load_network(
                load_folder_path, reconstruction_net, reconstruction_dict["load_path"]
            )
        except KeyError:
            print("No save_path found, loading default model")

        reconstruction_net.eval()
        ## Define sinogram transform
        sinogram_transforms = Normalise()

    display_transform = Normalise()

    for epoch in range(training_dict["n_epochs"]):
        print(f"Training epoch {epoch} / {training_dict['n_epochs']}...")
        for index, data in enumerate(train_dataloader):
            reconstruction = data[0].to(segmentation_device)
            mask = data[-1].to(segmentation_device)
            if training_dict["reconstructed"]:
                with torch.no_grad():
                    ## Re-sample
                    sinogram = odl_backend.get_sinogram(reconstruction)
                    sinogram = sinogram_transforms(sinogram)  # type:ignore
                    ## Reconstruct
                    if reconstruction_dict["name"] == "lpd":  # type:ignore
                        reconstruction = reconstruction_net(sinogram, just_infer=True)  # type:ignore
                    elif reconstruction_dict["name"] == "fourier_filtering":  # type:ignore
                        filtered_sinogram: torch.Tensor = reconstruction_net(sinogram)  # type:ignore
                        reconstruction = odl_backend.get_reconstruction(
                            filtered_sinogram.unsqueeze(1)
                        )
                    else:
                        raise NotImplementedError
            optimiser.zero_grad()
            approximated_segmentation = segmentation_net(
                reconstruction * mask[:, 1:, :, :]
            )
            loss_segmentation = segmentation_loss(
                approximated_segmentation, mask[:, 1:, :, :]
            )

            loss_segmentation.backward()

            optimiser.step()

            if index % 10 == 0:
                if verbose:
                    print(f"\n Metrics at step {index} of epoch {epoch}")
                    print(f"Image BCE Loss : {loss_segmentation.item()}")
                run_writer.add_scalar(
                    f"Image BCE Loss",
                    loss_segmentation.item(),
                    global_step=index + epoch * train_dataloader.__len__(),
                )
                # image_writer.write_image_tensor(reconstruction, 'current_reconstruction.jpg')
                image_writer.write_image_tensor(
                    torch.cat(
                        (
                            display_transform(reconstruction[0, 0]),
                            display_transform(approximated_segmentation[0, 0]),
                            display_transform(mask[0, 1]),
                        ),
                        dim=1,
                    ),
                    "input_segmentation_tgt.jpg",
                )

        torch.save(segmentation_net.state_dict(), save_file_path)
