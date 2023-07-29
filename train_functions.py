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
from evaluate import get_inference_function


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
            elif network_name == 'fourier_filtering':
                network = FourierFilteringModule(
                    network_dict,
                    n_measurements=odl_backend.angle_partition_dict["shape"], # type:ignore
                    detector_size=odl_backend.detector_partition_dict["shape"], # type:ignore
                    device=current_device
                    )
            elif network_name =='filtered_backprojection':
                network = None

            else:
                raise NotImplementedError(f"{network_name} not implemented")

        elif architecture_name =='segmentation':

            if network_name =='Unet':
                network = Unet(network_dict['unet_dict']).to(current_device)

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
    metadata_dict: Dict,
    train_dataloader: DataLoader,
    image_writer: PyPlotImageWriter,
    run_writer: SummaryWriter,
    load_folder_path:pathlib.Path,
    save_file_path: pathlib.Path,
    verbose=True,
):
    architecture_dict = metadata_dict['architecture_dict']
    training_dict=metadata_dict['training_dict']
    data_feeding_dict=metadata_dict['data_feeding_dict']

    segmentation_dict = architecture_dict["segmentation"]
    segmentation_device = torch.device(segmentation_dict["device_name"])

    networks = unpack_architecture_dicts(architecture_dict, odl_backend)
    segmentation_net = networks['segmentation']
    segmentation_net = load_network(load_folder_path, segmentation_net, segmentation_dict["load_path"])

    segmentation_loss = loss_name_to_loss_function(training_dict["segmentation_loss"])

    optimiser = torch.optim.Adam(
        params=segmentation_net.parameters(),
        lr=training_dict["learning_rate"],
        betas=(0.9, 0.99),
    )

    if data_feeding_dict["reconstructed"]:
        '''reconstruction_dict = architecture_dict["reconstruction"]
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
        '''
        reconstruction_dict = architecture_dict["reconstruction"]
        model_load_path = ""
        if "load_path" in reconstruction_dict.keys():
            model_load_path = reconstruction_dict["load_path"]

        inference_function = get_inference_function(
            metadata_dict,
            'reconstruction',
            odl_backend,
            load_folder_path,
            model_load_path)

        sinogram_transforms = Normalise()

    if segmentation_dict['folded']:
        unfold = torch.nn.Unfold((64,64), stride=64)
        fold = torch.nn.Fold(torch.Size([512,512]), 64,stride=64)

    display_transform = Normalise()

    for epoch in range(training_dict["n_epochs"]):
        print(f"Training epoch {epoch} / {training_dict['n_epochs']}...")
        for index, data in enumerate(train_dataloader):
            reconstruction = data[0].to(segmentation_device)
            mask = data[-1].to(segmentation_device)
            if data_feeding_dict["reconstructed"]:
                with torch.no_grad():
                    ## Re-sample
                    sinogram = odl_backend.get_sinogram(reconstruction)
                    sinogram = sinogram_transforms(sinogram)  # type:ignore
                    ## Reconstruct
                    reconstruction = inference_function(sinogram) # type:ignore

                    if reconstruction_dict['name'] == 'fourier_filtering': # type:ignore
                        reconstruction = odl_backend.get_reconstruction(reconstruction)
                    elif reconstruction_dict['name'] == 'lpd': # type:ignore
                        reconstruction = reconstruction[0]

            optimiser.zero_grad()
            if segmentation_dict['folded']:
                reconstruction:torch.Tensor = unfold(reconstruction) #type:ignore
                reconstruction = reconstruction.transpose(1, 2).view(data_feeding_dict['batch_size'], 64, 64, 64)

            approximated_segmentation:torch.Tensor = segmentation_net(reconstruction)

            if segmentation_dict['folded']:
                    approximated_segmentation = fold(approximated_segmentation.reshape([data_feeding_dict['batch_size'], 64,4096]).transpose(1,2)) #type:ignore

            if segmentation_dict['output_tensor']   == "reconstruction":
                loss_segmentation = segmentation_loss(approximated_segmentation, reconstruction)
            elif segmentation_dict['output_tensor'] == "mask":
                loss_segmentation = segmentation_loss(approximated_segmentation, mask[:,1:,:,:])
            elif segmentation_dict['output_tensor'] == "background_mask":
                loss_segmentation = segmentation_loss(approximated_segmentation, mask)
            elif segmentation_dict['output_tensor'] == "background_mask_reconstruction":
                loss_segmentation = segmentation_loss(approximated_segmentation, torch.cat([mask, reconstruction], dim=1))
            else:
                print(f'Ouput tensor {training_dict["output_tensor"]} not implemented')
                raise NotImplementedError

            loss_segmentation.backward()
            optimiser.step()

            if index % 50 == 0:
                if verbose:
                    print(f"\n Metrics at step {index} of epoch {epoch}")
                    print(f"Image BCE Loss : {loss_segmentation.item()}")
                run_writer.add_scalar(
                    f"Image BCE Loss",
                    loss_segmentation.item(),
                    global_step=index + epoch * train_dataloader.__len__(),
                )
                if segmentation_dict['folded']:
                    reconstruction = fold(reconstruction.reshape([data_feeding_dict['batch_size'], 64,4096]).transpose(1,2)) #type:ignore
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

def train_joint_pipeline(
    odl_backend: ODLBackend,
    metadata_dict: Dict,
    train_dataloader: DataLoader,
    image_writer: PyPlotImageWriter,
    run_writer: SummaryWriter,
    load_folder_path:pathlib.Path,
    save_file_path: pathlib.Path,
    verbose=True
):
    architecture_dict = metadata_dict['architecture_dict']
    training_dict=metadata_dict['training_dict']
    data_feeding_dict=metadata_dict['data_feeding_dict']

    networks = unpack_architecture_dicts(architecture_dict, odl_backend)

    segmentation_dict = architecture_dict["segmentation"]
    segmentation_device = torch.device(segmentation_dict["device_name"])

    segmentation_net = networks['segmentation']
    segmentation_net = load_network(load_folder_path, segmentation_net, segmentation_dict["load_path"])

    reconstruction_dict = architecture_dict["reconstruction"]
    reconstruction_device = torch.device(reconstruction_dict["device_name"])

    reconstruction_net = networks['reconstruction']
    reconstruction_net = load_network(load_folder_path, reconstruction_net, reconstruction_dict["load_path"])

    segmentation_loss   = loss_name_to_loss_function(training_dict["segmentation_loss"])
    reconstruction_loss = loss_name_to_loss_function(training_dict["reconstruction_loss"])
    sinogram_loss = loss_name_to_loss_function(training_dict["sinogram_loss"])

    optimiser = torch.optim.Adam(
        params=list(reconstruction_net.parameters()) + list(segmentation_net.parameters()),
        lr=training_dict["learning_rate"],
        betas=(0.9, 0.99),
    )

    sinogram_transforms = Normalise()
    display_transform = Normalise()

    psnr_loss = PSNR()

    for epoch in range(training_dict["n_epochs"]):
        print(f"Training epoch {epoch} / {training_dict['n_epochs']}...")
        for index, data in enumerate(train_dataloader):
            reconstruction = data[0].to(segmentation_device)
            mask = data[-1].to(segmentation_device)

            ## Re-sample
            sinogram = odl_backend.get_sinogram(reconstruction)
            sinogram = sinogram_transforms(sinogram)  # type:ignore

            optimiser.zero_grad()
            ## Reconstruct
            approximated_reconstruction, approximated_sinogram = reconstruction_net(sinogram)
            loss_recontruction = reconstruction_loss(approximated_reconstruction, reconstruction)
            loss_sinogram = sinogram_loss(approximated_sinogram, sinogram)
            total_reconstruction_loss = (1 - training_dict["dual_loss_weighting"]) * loss_recontruction + training_dict["dual_loss_weighting"] * loss_sinogram

            approximated_segmentation:torch.Tensor = segmentation_net(approximated_reconstruction)
            loss_segmentation = segmentation_loss(approximated_segmentation, mask[:,1:,:,:])

            total_loss = training_dict['segmentation_loss_weighting']*loss_segmentation + (1-training_dict['segmentation_loss_weighting'])*total_reconstruction_loss

            total_loss.backward()
            optimiser.step()

            if index % 50 == 0:
                if verbose:
                    print(f"\n Metrics at step {index} of epoch {epoch}")

                    print(f"Primal PSNR : {psnr_loss(approximated_reconstruction, reconstruction).item()}")
                    print(f"Dual PSNR : {psnr_loss(approximated_sinogram, sinogram).item()}")
                    print(f"Total reconstruction loss : {total_reconstruction_loss.item()}")
                    print(f"Image BCE Loss : {loss_segmentation.item()}")
                    print(f"Total loss : {total_loss.item()}")


                run_writer.add_scalar(f"Primal PSNR", psnr_loss(approximated_reconstruction, reconstruction).item(),global_step=index + epoch * train_dataloader.__len__())
                run_writer.add_scalar(f"Dual PSNR", psnr_loss(approximated_sinogram, sinogram).item(),global_step=index + epoch * train_dataloader.__len__())
                run_writer.add_scalar(f"Total reconstruction loss", total_reconstruction_loss.item(),global_step=index + epoch * train_dataloader.__len__())
                run_writer.add_scalar(f"Image BCE Loss", loss_segmentation.item(),global_step=index + epoch * train_dataloader.__len__())
                run_writer.add_scalar(f"Total Loss", total_loss.item(),global_step=index + epoch * train_dataloader.__len__())

                targets = torch.cat(
                        (
                            display_transform(reconstruction[0, 0]),
                            display_transform(mask[0, 1]),
                        ),  dim=1)

                approxs = torch.cat(
                        (
                            display_transform(approximated_reconstruction[0, 0]),
                            display_transform(approximated_segmentation[0, 0]),
                        ),  dim=1)


                image_writer.write_image_tensor(torch.cat((targets,approxs),dim=0,),"input_segmentation_tgt.jpg")

        torch.save({
            'reconstruction_net': reconstruction_net.state_dict(),
            'segmentation_net': segmentation_net.state_dict(),
            }, save_file_path)


