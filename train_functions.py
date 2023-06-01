from typing import Dict
import pathlib

from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter #type:ignore
from torchvision.transforms import Compose

from models import LearnedPrimalDual, Unet2D512 #type:ignore
from backends.odl import ODLBackend
from transforms import PoissonSinogramTransform #type:ignore

def normalise(x:torch.Tensor) -> torch.Tensor:
    return( x-x.min()) / (x.max() - x.min())

def load_network(network:torch.nn.Module, load_path:pathlib.Path ):
    load_path = pathlib.Path(load_path)
    if load_path.is_file():
        print(f'Loading model state_dict from {load_path}')
        network.load_state_dict(torch.load(load_path))
    else:
        print(f'No file found at {load_path}, no initialisation')
    return network

def train_joint(
        dimension:int,
        odl_backend:ODLBackend,
        architecture_dict:Dict,
        training_dict:Dict,
        train_dataloader:DataLoader,
        test_dataloader:DataLoader
        ):

    reconstruction_device = torch.device(architecture_dict['reconstruction']['device_name'])
    segmentation_device = torch.device(architecture_dict['segmentation']['device_name'])

    reconstruction_net = LearnedPrimalDual(
        dimension = dimension,
        odl_backend = odl_backend,
        n_iterations = architecture_dict['reconstruction']['lpd_n_iterations'],
        n_filters = architecture_dict['reconstruction']['lpd_n_filters'],
        device = reconstruction_device
        )

    segmentation_net = Unet2D512(
        input_channels  = architecture_dict['segmentation']['Unet_input_channels'],
        output_channels = architecture_dict['segmentation']['Unet_output_channels'],
        n_filters = architecture_dict['segmentation']['Unet_n_filters']
        ).to(segmentation_device)

    reconstruction_net = load_network(reconstruction_net, architecture_dict['reconstruction']['load_path'])
    segmentation_net = load_network(segmentation_net, architecture_dict['segmentation']['load_path'])

    reconstruction_loss = torch.nn.MSELoss()
    segmentation_loss   = torch.nn.BCELoss()

    optimiser = torch.optim.Adam(
        params = list(reconstruction_net.parameters()) + list(segmentation_net.parameters()),
        lr = training_dict['learning_rate'],
        betas=(0.9,0.99)
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer = optimiser,
        T_max = training_dict['n_epochs']
    )

    pathlib.Path(f'runs/joint/{training_dict["C"]:.4f}').mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(
        log_dir = f'runs/joint/{training_dict["C"]:.4f}'
    )

    max_rec_loss = 1e8
    max_seg_loss = 1e8
    max_tot_loss = max_rec_loss + max_seg_loss + 1

    # Wrap models with DataParallel or DistributedDataParallel
    reconstruction_net = torch.nn.parallel.DistributedDataParallel(reconstruction_net, device_ids=[reconstruction_device, segmentation_device])
    segmentation_net = torch.nn.parallel.DistributedDataParallel(segmentation_net, device_ids=[reconstruction_device, segmentation_device])

    # Transforms
    if dimension ==1:
        sinogram_size = [training_dict['batch_size'], odl_backend.angle_partition_dict['shape'], odl_backend.detector_partition_dict['shape']]
    else:
        sinogram_size = [training_dict['batch_size'], 1, odl_backend.angle_partition_dict['shape'], odl_backend.detector_partition_dict['shape']]

    sinogram_transforms = PoissonSinogramTransform(
        I0 = 100*training_dict['dose_percent'],
        device=reconstruction_device,
        sinogram_size=torch.Size(sinogram_size),
    )

    for epoch in range(training_dict['n_epochs']):
        print(f"Training epoch {epoch} / {training_dict['n_epochs']}...")
        for index, (reconstruction, mask) in enumerate(tqdm(train_dataloader)):
            reconstruction = reconstruction.to(reconstruction_device)

            mask = mask.to(segmentation_device)
            optimiser.zero_grad()
            sinogram = odl_backend.get_sinogram(reconstruction)

            if dimension == 1:
                sinogram = torch.squeeze(sinogram)

            sinogram = sinogram_transforms(sinogram)['sinogram']

            approximated_reconstruction = reconstruction_net(sinogram).to(segmentation_device)

            loss_recontruction = reconstruction_loss(approximated_reconstruction, reconstruction)

            approximated_segmentation = segmentation_net(approximated_reconstruction)

            loss_segmentation  = segmentation_loss(approximated_segmentation, mask)

            loss_total = (1-training_dict['C'])*loss_recontruction + training_dict['C']*loss_segmentation

            loss_total.backward()
            torch.nn.utils.clip_grad.clip_grad_norm_(
                parameters=reconstruction_net.parameters(),
                max_norm=1.0,
                norm_type=2
                )

            optimiser.step()
            scheduler.step()

            print(f'Reconstruction Loss : {loss_recontruction.item():.5f}')
            print(f'Segmentation Loss : {loss_segmentation.item():.5f}')
            print(f'Weighted C = {training_dict["C"]} Total Loss {loss_total.item():.5f}')

            writer.add_scalar('Reconstruction Loss', loss_recontruction.item())
            writer.add_scalar('Segmentation Loss', loss_segmentation.item())
            writer.add_scalar(f'Weighted C = {training_dict["C"]} Total Loss', loss_total.item())

        print(f'Evaluating on test_dataset... ')
        reconstruction_net.eval()
        segmentation_net.eval()
        test_loss_reconstruction = 0
        test_loss_segmentation = 0
        test_loss_total = 0
        for index, (reconstruction, mask) in enumerate(tqdm(test_dataloader)):
            sinogram = odl_backend.get_sinogram(reconstruction)
            approximated_reconstruction = reconstruction_net(sinogram)
            loss_recontruction = reconstruction_loss(approximated_reconstruction, reconstruction)
            approximated_segmentation = segmentation_net(approximated_reconstruction)
            loss_segmentation  = segmentation_loss(approximated_segmentation, mask)
            loss_total = (1-training_dict['C'])*loss_recontruction + training_dict['C']*loss_segmentation
            test_loss_reconstruction += loss_recontruction.item()
            test_loss_segmentation += loss_segmentation.item()
            test_loss_total += (1-training_dict['C'])*loss_recontruction.item() + training_dict['C']*loss_segmentation.item()

        print(f'Test Reconstruction Loss : {test_loss_reconstruction:.5f}')
        print(f'Test Segmentation Loss : {test_loss_segmentation:.5f}')
        print(f'Weighted C = {training_dict["C"]} Test Total Loss {test_loss_total:.5f}')

        writer.add_scalar('Reconstruction Loss on Test Set', test_loss_reconstruction)
        writer.add_scalar('Segmentation Loss on Test Set', test_loss_segmentation)
        writer.add_scalar(f'Weighted C = {training_dict["C"]} Total Loss on Test Set', test_loss_total)
        writer.add_image(f'Reconstruction at step {epoch}', approximated_reconstruction[0,0], dataformats='HW') #type:ignore
        writer.add_image(f'Segmentation at step {epoch}', approximated_segmentation[0,0], dataformats='HW') #type:ignore

        if test_loss_total <= max_tot_loss:
            reconstruction_model_save_path = pathlib.Path(architecture_dict['reconstruction']['save_path'])
            reconstruction_model_save_path.parent.mkdir(exist_ok=True, parents=True)
            segmentation_model_save_path = pathlib.Path(architecture_dict['segmentation']['save_path'])
            segmentation_model_save_path.parent.mkdir(exist_ok=True, parents=True)

            torch.save(reconstruction_net.state_dict(), reconstruction_model_save_path)
            torch.save(segmentation_net.state_dict(), segmentation_model_save_path)

        reconstruction_net.train()
        segmentation_net.train()

    return 0

def train_end_to_end(
        dimension:int,
        odl_backend:ODLBackend,
        architecture_dict:Dict,
        training_dict:Dict,
        train_dataloader:DataLoader,
        test_dataloader:DataLoader
        ):

    reconstruction_device = torch.device(architecture_dict['reconstruction']['device_name'])
    segmentation_device = torch.device(architecture_dict['segmentation']['device_name'])

    reconstruction_net = LearnedPrimalDual(
        dimension = dimension,
        odl_backend = odl_backend,
        n_iterations = architecture_dict['reconstruction']['lpd_n_iterations'],
        n_filters = architecture_dict['reconstruction']['lpd_n_filters'],
        device = reconstruction_device
        )

    segmentation_net = Unet2D512(
        input_channels  = architecture_dict['segmentation']['Unet_input_channels'],
        output_channels = architecture_dict['segmentation']['Unet_output_channels'],
        n_filters = architecture_dict['segmentation']['Unet_n_filters']
        ).to(segmentation_device)

    reconstruction_net = load_network(reconstruction_net, architecture_dict['reconstruction']['load_path'])
    segmentation_net = load_network(segmentation_net, architecture_dict['segmentation']['load_path'])

    reconstruction_loss = torch.nn.MSELoss()
    segmentation_loss   = torch.nn.BCELoss()

    optimiser = torch.optim.Adam(
        params = list(reconstruction_net.parameters()) + list(segmentation_net.parameters()),
        lr = training_dict['learning_rate'],
        betas=(0.9,0.99)
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer = optimiser,
        T_max = training_dict['n_epochs']
    )

    pathlib.Path(f'runs/end_to_end').mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(
        log_dir = f'runs/end_to_end'
    )

    max_rec_loss = 1e8
    max_seg_loss = 1e8
    max_tot_loss = max_rec_loss + max_seg_loss + 1

    # Wrap models with DataParallel or DistributedDataParallel
    reconstruction_net = torch.nn.parallel.DistributedDataParallel(reconstruction_net, device_ids=[reconstruction_device, segmentation_device])
    segmentation_net = torch.nn.parallel.DistributedDataParallel(segmentation_net, device_ids=[reconstruction_device, segmentation_device])

    # Transforms
    if dimension ==1:
        sinogram_size = [training_dict['batch_size'], odl_backend.angle_partition_dict['shape'], odl_backend.detector_partition_dict['shape']]
    else:
        sinogram_size = [training_dict['batch_size'], 1, odl_backend.angle_partition_dict['shape'], odl_backend.detector_partition_dict['shape']]

    sinogram_transforms = PoissonSinogramTransform(
        I0 = 100*training_dict['dose_percent'],
        device=reconstruction_device,
        sinogram_size=torch.Size(sinogram_size),
    )

    for epoch in range(training_dict['n_epochs']):
        print(f"Training epoch {epoch} / {training_dict['n_epochs']}...")
        for index, (reconstruction, mask) in enumerate(tqdm(train_dataloader)):
            reconstruction = reconstruction.to(reconstruction_device)
            mask = mask.to(segmentation_device)
            optimiser.zero_grad()
            sinogram = odl_backend.get_sinogram(reconstruction)

            if dimension == 1:
                sinogram = torch.squeeze(sinogram)

            sinogram = sinogram_transforms(sinogram)['sinogram']

            approximated_reconstruction = reconstruction_net(sinogram).to(segmentation_device)

            loss_recontruction = reconstruction_loss(approximated_reconstruction, reconstruction)

            approximated_segmentation = segmentation_net(approximated_reconstruction)

            loss_segmentation  = segmentation_loss(approximated_segmentation, mask)

            loss_total = loss_recontruction + loss_segmentation

            loss_segmentation.backward()
            torch.nn.utils.clip_grad.clip_grad_norm_(
                parameters=reconstruction_net.parameters(),
                max_norm=1.0,
                norm_type=2
                )

            optimiser.step()
            scheduler.step()

            print(f'Reconstruction Loss : {loss_recontruction.item():.5f}')
            print(f'Segmentation Loss : {loss_segmentation.item():.5f}')
            print(f'Total Loss : {loss_total.item():.5f}')

            writer.add_scalar('Reconstruction Loss', loss_recontruction.item())
            writer.add_scalar('Segmentation Loss', loss_segmentation.item())
            writer.add_scalar('Total Loss', loss_total.item())

        print(f'Evaluating on test_dataset... ')
        reconstruction_net.eval()
        segmentation_net.eval()
        test_loss_reconstruction = 0
        test_loss_segmentation = 0
        test_loss_total = 0
        for index, (reconstruction, mask) in enumerate(tqdm(test_dataloader)):
            reconstruction = reconstruction.to(reconstruction_device)
            sinogram = odl_backend.get_sinogram(reconstruction)
            approximated_reconstruction = reconstruction_net(sinogram)
            loss_recontruction = reconstruction_loss(approximated_reconstruction, reconstruction)
            approximated_segmentation = segmentation_net(approximated_reconstruction)
            loss_segmentation  = segmentation_loss(approximated_segmentation, mask.to(reconstruction_device))
            loss_total = loss_recontruction + loss_segmentation
            test_loss_reconstruction += loss_recontruction.item()
            test_loss_segmentation += loss_segmentation.item()
            test_loss_total += loss_recontruction.item() +loss_segmentation.item()

        print(f'Test Reconstruction Loss : {test_loss_reconstruction:.5f}')
        print(f'Test Segmentation Loss : {test_loss_segmentation:.5f}')

        writer.add_scalar('Reconstruction Loss on Test Set', test_loss_reconstruction)
        writer.add_scalar('Segmentation Loss on Test Set', test_loss_segmentation)
        writer.add_image(f'Reconstruction at step {epoch}', approximated_reconstruction[0,0], dataformats='HW') #type:ignore
        writer.add_image(f'Segmentation at step {epoch}', approximated_segmentation[0,0], dataformats='HW') #type:ignore

        if test_loss_total <= max_tot_loss:
            reconstruction_model_save_path = pathlib.Path(architecture_dict['reconstruction']['save_path'])
            reconstruction_model_save_path.parent.mkdir(exist_ok=True, parents=True)
            segmentation_model_save_path = pathlib.Path(architecture_dict['segmentation']['save_path'])
            segmentation_model_save_path.parent.mkdir(exist_ok=True, parents=True)

            torch.save(reconstruction_net.state_dict(), reconstruction_model_save_path)
            torch.save(segmentation_net.state_dict(), segmentation_model_save_path)

        reconstruction_net.train()
        segmentation_net.train()

    return 0

def train_reconstruction_network(
        dimension:int,
        odl_backend:ODLBackend,
        architecture_dict:Dict,
        training_dict:Dict,
        train_dataloader:DataLoader,
        test_dataloader:DataLoader
        ):

    reconstruction_device = torch.device(architecture_dict['reconstruction']['device_name'])

    reconstruction_net = LearnedPrimalDual(
        dimension = dimension,
        odl_backend = odl_backend,
        n_iterations = architecture_dict['reconstruction']['lpd_n_iterations'],
        n_filters = architecture_dict['reconstruction']['lpd_n_filters'],
        device = reconstruction_device
        )

    reconstruction_net = load_network(reconstruction_net, architecture_dict['reconstruction']['load_path'])

    reconstruction_net = torch.compile(reconstruction_net)

    reconstruction_loss = torch.nn.MSELoss()

    optimiser = torch.optim.Adam(
        params = reconstruction_net.parameters(),
        lr = training_dict['learning_rate'],
        betas=(0.9,0.99)
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer = optimiser,
        T_max = training_dict['n_epochs']
    )

    pathlib.Path(f'runs/reconstruction').mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(
        log_dir = f'runs/reconstruction'
    )

    max_rec_loss = 1e8

    # Transforms
    if dimension ==1:
        sinogram_size = [training_dict['batch_size'], odl_backend.angle_partition_dict['shape'], odl_backend.detector_partition_dict['shape']]
    else:
        sinogram_size = [training_dict['batch_size'], 1, odl_backend.angle_partition_dict['shape'], odl_backend.detector_partition_dict['shape']]

    sinogram_transforms = PoissonSinogramTransform(
        I0 = 100*training_dict['dose_percent'],
        device=reconstruction_device,
        sinogram_size=torch.Size(sinogram_size),
    )

    for epoch in range(training_dict['n_epochs']):
        print(f"Training epoch {epoch} / {training_dict['n_epochs']}...")
        for index, reconstruction in enumerate(tqdm(train_dataloader)):
            reconstruction = reconstruction.to(reconstruction_device)
            optimiser.zero_grad()
            sinogram = odl_backend.get_sinogram(reconstruction)

            if dimension == 1:
                sinogram = torch.squeeze(sinogram)

            sinogram = sinogram_transforms(sinogram)['sinogram']

            approximated_reconstruction = reconstruction_net(sinogram)

            loss_recontruction = reconstruction_loss(approximated_reconstruction, reconstruction)

            loss_recontruction.backward()
            torch.nn.utils.clip_grad.clip_grad_norm_(
                parameters=reconstruction_net.parameters(),
                max_norm=1.0,
                norm_type=2
                )

            optimiser.step()
            scheduler.step()

            print(f'Reconstruction Loss : {loss_recontruction.item():.5f}')

            writer.add_scalar('Reconstruction Loss', loss_recontruction.item())

        print(f'Evaluating on test_dataset... ')
        reconstruction_net.eval()
        test_loss_reconstruction = 0
        for index, reconstruction in enumerate(tqdm(test_dataloader)):
            reconstruction = reconstruction.to(reconstruction_device)
            sinogram = odl_backend.get_sinogram(reconstruction)
            approximated_reconstruction:torch.Tensor = reconstruction_net(sinogram)
            loss_recontruction = reconstruction_loss(approximated_reconstruction, reconstruction)
            test_loss_reconstruction += loss_recontruction.item()

        print(f'Test Reconstruction Loss : {test_loss_reconstruction:.5f}')
        writer.add_scalar('Reconstruction Loss on Test Set', test_loss_reconstruction)

        writer.add_image(f'Reconstruction at step {epoch}', approximated_reconstruction[0,0].detach().cpu(), dataformats='HW') #type:ignore

        if test_loss_reconstruction <= max_rec_loss:
            reconstruction_model_save_path = pathlib.Path(architecture_dict['reconstruction']['save_path'])
            reconstruction_model_save_path.parent.mkdir(exist_ok=True, parents=True)
            torch.save(reconstruction_net.state_dict(), reconstruction_model_save_path)
        reconstruction_net.train()

    print('Training Finished \u2713 ')

def train_segmentation_network(
        dimension:int,
        architecture_dict:Dict,
        training_dict:Dict,
        train_dataloader:DataLoader,
        test_dataloader:DataLoader
        ):

    segmentation_device = torch.device(architecture_dict['segmentation']['device_name'])

    segmentation_net = Unet2D512(
        input_channels  = architecture_dict['segmentation']['Unet_input_channels'],
        output_channels = architecture_dict['segmentation']['Unet_output_channels'],
        n_filters = architecture_dict['segmentation']['Unet_n_filters']
        ).to(segmentation_device)

    segmentation_net = load_network(segmentation_net, architecture_dict['segmentation']['load_path'])

    segmentation_loss   = torch.nn.BCELoss()

    optimiser = torch.optim.Adam(
        params = segmentation_net.parameters(),
        lr = training_dict['learning_rate'],
        betas=(0.9,0.99)
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer = optimiser,
        T_max = training_dict['n_epochs']
    )

    pathlib.Path(f'runs/segmentation').mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(
        log_dir = f'runs/segmentation'
    )

    max_seg_loss = 1e8

    for epoch in range(training_dict['n_epochs']):
        print(f"Training epoch {epoch} / {training_dict['n_epochs']}...")
        for index, (reconstruction, mask) in enumerate(tqdm(train_dataloader)):
            reconstruction = reconstruction.to(segmentation_device)
            mask = mask.to(segmentation_device)
            optimiser.zero_grad()
            approximated_segmentation = segmentation_net(reconstruction)
            loss_segmentation  = segmentation_loss(approximated_segmentation, mask)

            loss_segmentation.backward()
            torch.nn.utils.clip_grad.clip_grad_norm_(
                parameters=segmentation_net.parameters(),
                max_norm=1.0,
                norm_type=2
                )

            optimiser.step()
            scheduler.step()

            print(f'Segmentation Loss : {loss_segmentation.item():.5f}')

            writer.add_scalar('Segmentation Loss', loss_segmentation.item())

        print(f'Evaluating on test_dataset... ')
        segmentation_net.eval()
        test_loss_segmentation = 0
        for index, (reconstruction, mask) in enumerate(tqdm(test_dataloader)):
            reconstruction = reconstruction.to(segmentation_device)
            mask = mask.to(segmentation_device)
            optimiser.zero_grad()
            approximated_segmentation = segmentation_net(reconstruction)
            loss_segmentation  = segmentation_loss(approximated_segmentation, mask)
            test_loss_segmentation += loss_segmentation.item()

        print(f'Test Segmentation Loss : {test_loss_segmentation:.5f}')

        writer.add_scalar('Segmentation Loss on Test Set', test_loss_segmentation)
        writer.add_image(f'Segmentation at step {epoch}', approximated_segmentation[0,0], dataformats='HW') #type:ignore

        if test_loss_segmentation <= max_seg_loss:
            segmentation_model_save_path = pathlib.Path(architecture_dict['segmentation']['save_path'])
            segmentation_model_save_path.parent.mkdir(exist_ok=True, parents=True)
            torch.save(segmentation_net.state_dict(), segmentation_model_save_path)

        segmentation_net.train()

    return 0