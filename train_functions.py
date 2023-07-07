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

def train_fbp(
        dimension:int,
        odl_backend:ODLBackend,
        architecture_dict:Dict,
        training_dict:Dict,
        angle_partition_dict:Dict,
        train_dataloader:DataLoader,
        image_writer:PyPlotImageWriter
        ):

    reconstruction_device = torch.device(architecture_dict['fourier_filter']['device_name'])

    fourier_filtering_module = FourierFilteringModule(dimension, angle_partition_dict['shape'], architecture_dict['fourier_filter']['n_filters'], reconstruction_device, 'custom', True)

    fourier_filtering_module = load_network(fourier_filtering_module, architecture_dict['fourier_filter']['load_path'])

    reconstruction_loss = torch.nn.MSELoss()
    psnr_loss = PSNR()

    optimiser = torch.optim.Adam(
        params = fourier_filtering_module.parameters(),
        lr = training_dict['learning_rate'],
        betas=(0.9,0.99)
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer = optimiser,
        T_max = training_dict['n_epochs']
    )

    pathlib.Path(f'/local/scratch/public/ev373/runs/fourier_filtering').mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(
        log_dir = f'/local/scratch/public/ev373/runs/fourier_filtering'
    )

    pathlib.Path('images/reconstruction').mkdir(parents=True, exist_ok=True)

    max_rec_loss = 1e8

    # Transforms
    if dimension ==1:
        sinogram_size = [training_dict['batch_size'], odl_backend.angle_partition_dict['shape'], odl_backend.detector_partition_dict['shape']]
    else:
        sinogram_size = [training_dict['batch_size'], 1, odl_backend.angle_partition_dict['shape'], odl_backend.detector_partition_dict['shape']]

    '''sinogram_transforms = PoissonSinogramTransform(
        I0 = 100*training_dict['dose'],
        device=reconstruction_device,
        sinogram_size=torch.Size(sinogram_size),
    )'''


    sinogram_transforms = Normalise()
    for epoch in range(training_dict['n_epochs']):
        print(f"Training epoch {epoch} / {training_dict['n_epochs']}...")
        for index, reconstruction in enumerate(tqdm(train_dataloader)):

            reconstruction = reconstruction.to(reconstruction_device)
            optimiser.zero_grad()
            sinogram = odl_backend.get_sinogram(reconstruction)

            if dimension == 1:
                sinogram = torch.squeeze(sinogram, dim=1)

            sinogram = sinogram_transforms(sinogram)

            approximated_reconstruction = fourier_filtering_module(sinogram).unsqueeze(1)

            loss_recontruction = reconstruction_loss(approximated_reconstruction, reconstruction)

            total_loss = loss_recontruction

            total_loss.backward()
            torch.nn.utils.clip_grad.clip_grad_norm_(
                parameters=fourier_filtering_module.parameters(),
                max_norm=1.0,
                norm_type=2
                )

            optimiser.step()
            scheduler.step()

            if index %10 == 0:
                backprojection = odl_backend.get_reconstruction(sinogram.unsqueeze(1))

                print(f'\n Metrics at step {index} of epoch {epoch}')
                print(f'Image MSE : {loss_recontruction.item()}')
                print(f'Image PSNR : {psnr_loss(approximated_reconstruction, reconstruction).item()}')
                print(f'FBP PSNR : {psnr_loss(backprojection, reconstruction).item()}')
                writer.add_scalar('Image Reconstruction Loss', loss_recontruction.item(), global_step=index+epoch*train_dataloader.__len__())
                writer.add_scalar('Image PSNR Loss', psnr_loss(approximated_reconstruction, reconstruction).item(), global_step=index+epoch*train_dataloader.__len__())

                #writer.add_image('Reconstruction', approximated_reconstruction[0,0].detach().cpu(), dataformats='HW')
                image_writer.write_image_tensor(approximated_reconstruction, 'current_reconstruction.jpg')
                image_writer.write_image_tensor(reconstruction, 'reconstruction_target.jpg')
                image_writer.write_image_tensor(sinogram.unsqueeze(1), 'sinogram_target.jpg')
                image_writer.write_image_tensor(backprojection, 'back_projection.jpg')

                image_writer.write_line_tensor(torch.real(fourier_filtering_module.filter.weight), 'weight_real.jpg') #type:ignore
                image_writer.write_line_tensor(torch.imag(fourier_filtering_module.filter.weight), 'weight_imag.jpg') #type:ignore

                input()

            if index%1000 == 0:
                reconstruction_model_save_path = pathlib.Path(architecture_dict['fourier_filter']['save_path'])
                reconstruction_model_save_path.parent.mkdir(exist_ok=True, parents=True)
                torch.save(fourier_filtering_module.state_dict(), reconstruction_model_save_path)

    print('Training Finished \u2713 ')

def train_joint(
        dimension:int,
        odl_backend:ODLBackend,
        architecture_dict:Dict,
        training_dict:Dict,
        train_dataloader:DataLoader,
        test_dataloader:DataLoader,
        image_writer:PyPlotImageWriter
        ):

    reconstruction_device = torch.device(architecture_dict['reconstruction']['device_name'])
    segmentation_device = torch.device(architecture_dict['segmentation']['device_name'])

    reconstruction_net = LearnedPrimalDual(
        dimension = dimension,
        odl_backend = odl_backend,
        n_primal=architecture_dict['reconstruction']['n_primal'],
        n_dual=architecture_dict['reconstruction']['n_dual'],
        n_iterations = architecture_dict['reconstruction']['lpd_n_iterations'],
        n_filters_primal = architecture_dict['reconstruction']['lpd_n_filters_primal'],
        n_filters_dual = architecture_dict['reconstruction']['lpd_n_filters_dual'],
        fourier_filtering = architecture_dict['reconstruction']['fourier_filtering'],
        device = reconstruction_device
        )

    segmentation_net = Unet(
        dimension=dimension,
        n_channels_input  = architecture_dict['segmentation']['Unet_input_channels'],
        n_channels_output = architecture_dict['segmentation']['Unet_output_channels'],
        n_filters = architecture_dict['segmentation']['Unet_n_filters']
        ).to(segmentation_device)

    reconstruction_net = load_network(reconstruction_net, architecture_dict['reconstruction']['load_path'])
    segmentation_net = load_network(segmentation_net, architecture_dict['segmentation']['load_path'])

    reconstruction_loss = torch.nn.MSELoss()
    segmentation_loss   = torch.nn.BCELoss()
    psnr_loss = PSNR()

    optimiser = torch.optim.Adam(
        params = list(reconstruction_net.parameters()) + list(segmentation_net.parameters()),
        lr = training_dict['learning_rate'],
        betas=(0.9,0.99)
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer = optimiser,
        T_max = training_dict['n_epochs']
    )

    pathlib.Path(f'/local/scratch/public/ev373/runs/joint/{training_dict["C"]:.4f}').mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(
        log_dir = f'/local/scratch/public/ev373/runs/joint/{training_dict["C"]:.4f}'
    )
    pathlib.Path(f'images/joint').mkdir(parents=True, exist_ok=True)

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

    sinogram_transforms = Normalise()

    for epoch in range(training_dict['n_epochs']):
        print(f"Training epoch {epoch} / {training_dict['n_epochs']}...")
        for index, (reconstruction, mask) in enumerate(tqdm(train_dataloader)):
            reconstruction = reconstruction.to(reconstruction_device)

            mask = mask.to(segmentation_device)
            optimiser.zero_grad()
            sinogram = odl_backend.get_sinogram(reconstruction)

            if dimension == 1:
                sinogram = torch.squeeze(sinogram, dim=1)

            sinogram = sinogram_transforms(sinogram)

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

            if index %10 == 0:
                print(f'\n Metrics at step {index} of epoch {epoch}')
                print(f'MSE Reconstruction: {loss_recontruction.item()}')
                print(f'PSNR Reconstruction: {psnr_loss(approximated_reconstruction, reconstruction).item()}')
                print(f'MSE Segmentation: {loss_segmentation.item()}')
                print(f'PSNR Segmentation: {psnr_loss(approximated_segmentation, mask).item()}')

                writer.add_scalar('Reconstruction Loss', loss_recontruction.item(), global_step=index+epoch*train_dataloader.__len__())
                writer.add_scalar('PSNR Loss', psnr_loss(approximated_reconstruction, reconstruction).item(), global_step=index+epoch*train_dataloader.__len__())
                writer.add_scalar('Reconstruction Loss', loss_recontruction.item(), global_step=index+epoch*train_dataloader.__len__())
                writer.add_scalar('Segmentation Loss', loss_segmentation.item(), global_step=index+epoch*train_dataloader.__len__())
                writer.add_scalar(f'Weighted C = {training_dict["C"]} Total Loss', loss_total.item(), global_step=index+epoch*train_dataloader.__len__())

            if index %100 == 0:
                writer.add_image('Reconstruction', approximated_reconstruction[0,0].detach().cpu(), dataformats='HW')
                writer.add_image('Segmentation', approximated_segmentation[0,0].detach().cpu(), dataformats='HW')
                image_writer.write_image_tensor(approximated_reconstruction, 'current_reconstruction.jpg')
                image_writer.write_image_tensor(approximated_segmentation, 'current_segmentation.jpg')

            if (index !=0 and index%1000 == 0):
                reconstruction_model_save_path = pathlib.Path(architecture_dict['reconstruction']['save_path'])
                reconstruction_model_save_path.parent.mkdir(exist_ok=True, parents=True)
                torch.save(reconstruction_net.state_dict(), reconstruction_model_save_path)
                segmentation_model_save_path = pathlib.Path(architecture_dict['segmentation']['save_path'])
                segmentation_model_save_path.parent.mkdir(exist_ok=True, parents=True)
                torch.save(segmentation_net.state_dict(), segmentation_model_save_path)

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

        writer.add_scalar('Reconstruction Loss on Test Set', test_loss_reconstruction, global_step=epoch)
        writer.add_scalar('Segmentation Loss on Test Set', test_loss_segmentation, global_step=epoch)
        writer.add_scalar(f'Weighted C = {training_dict["C"]} Total Loss on Test Set', test_loss_total, global_step=epoch)
        writer.add_image(f'Reconstruction at step {epoch}', approximated_reconstruction[0,0], dataformats='HW', global_step=epoch) #type:ignore
        writer.add_image(f'Segmentation at step {epoch}', approximated_segmentation[0,0], dataformats='HW', global_step=epoch) #type:ignore

        if test_loss_total <= max_tot_loss:
            reconstruction_model_save_path = pathlib.Path(architecture_dict['reconstruction']['save_path'])
            reconstruction_model_save_path.parent.mkdir(exist_ok=True, parents=True)
            segmentation_model_save_path = pathlib.Path(architecture_dict['segmentation']['save_path'])
            segmentation_model_save_path.parent.mkdir(exist_ok=True, parents=True)

            torch.save(reconstruction_net.state_dict(), reconstruction_model_save_path)
            torch.save(segmentation_net.state_dict(), segmentation_model_save_path)

        reconstruction_net.train()
        segmentation_net.train()

    print('Training Finished \u2713 ')

def train_end_to_end(
        dimension:int,
        odl_backend:ODLBackend,
        architecture_dict:Dict,
        training_dict:Dict,
        train_dataloader:DataLoader,
        test_dataloader:DataLoader,
        image_writer:PyPlotImageWriter
        ):

    reconstruction_device = torch.device(architecture_dict['reconstruction']['device_name'])
    segmentation_device = torch.device(architecture_dict['segmentation']['device_name'])

    reconstruction_net = LearnedPrimalDual(
        dimension = dimension,
        odl_backend = odl_backend,
        n_primal=architecture_dict['reconstruction']['n_primal'],
        n_dual=architecture_dict['reconstruction']['n_dual'],
        n_iterations = architecture_dict['reconstruction']['lpd_n_iterations'],
        n_filters_primal = architecture_dict['reconstruction']['lpd_n_filters_primal'],
        n_filters_dual = architecture_dict['reconstruction']['lpd_n_filters_dual'],
        fourier_filtering = architecture_dict['reconstruction']['fourier_filtering'],
        device = reconstruction_device
        )

    segmentation_net = Unet(
        dimension=dimension,
        n_channels_input  = architecture_dict['segmentation']['Unet_input_channels'],
        n_channels_output = architecture_dict['segmentation']['Unet_output_channels'],
        n_filters = architecture_dict['segmentation']['Unet_n_filters']
        ).to(segmentation_device)

    reconstruction_net = load_network(reconstruction_net, architecture_dict['reconstruction']['load_path'])
    segmentation_net = load_network(segmentation_net, architecture_dict['segmentation']['load_path'])

    reconstruction_loss = torch.nn.MSELoss()
    segmentation_loss   = torch.nn.BCELoss()
    psnr_loss = PSNR()

    optimiser = torch.optim.Adam(
        params = list(reconstruction_net.parameters()) + list(segmentation_net.parameters()),
        lr = training_dict['learning_rate'],
        betas=(0.9,0.99)
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer = optimiser,
        T_max = training_dict['n_epochs']
    )

    pathlib.Path(f'/local/scratch/public/ev373/runs/end_to_end').mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(
        log_dir = f'/local/scratch/public/ev373/runs/end_to_end'
    )
    pathlib.Path(f'images/end_to_end').mkdir(parents=True, exist_ok=True)

    max_rec_loss = 1e8
    max_seg_loss = 1e8
    max_tot_loss = max_rec_loss + max_seg_loss + 1

    # Wrap models with DataParallel or DistributedDataParallel
    reconstruction_net = torch.nn.parallel.DistributedDataParallel(reconstruction_net, device_ids=[reconstruction_device, segmentation_device])
    segmentation_net = torch.nn.parallel.DistributedDataParallel(segmentation_net, device_ids=[reconstruction_device, segmentation_device])

    # Transforms
    '''if dimension ==1:
        sinogram_size = [training_dict['batch_size'], odl_backend.angle_partition_dict['shape'], odl_backend.detector_partition_dict['shape']]
    else:
        sinogram_size = [training_dict['batch_size'], 1, odl_backend.angle_partition_dict['shape'], odl_backend.detector_partition_dict['shape']]

    sinogram_transforms = PoissonSinogramTransform(
        I0 = 100*training_dict['dose'],
        device=reconstruction_device,
        sinogram_size=torch.Size(sinogram_size),
    )'''

    sinogram_transforms = Normalise()

    for epoch in range(training_dict['n_epochs']):
        print(f"Training epoch {epoch} / {training_dict['n_epochs']}...")
        for index, (reconstruction, mask) in enumerate(tqdm(train_dataloader)):
            reconstruction = reconstruction.to(reconstruction_device)
            mask = mask.to(segmentation_device)
            optimiser.zero_grad()
            sinogram = odl_backend.get_sinogram(reconstruction)

            if dimension == 1:
                sinogram = torch.squeeze(sinogram, dim=1)

            sinogram = sinogram_transforms(sinogram)

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

            if index %10 == 0:
                print(f'\n Metrics at step {index} of epoch {epoch}')
                print(f'MSE Reconstruction: {loss_recontruction.item()}')
                print(f'PSNR Reconstruction: {psnr_loss(approximated_reconstruction, reconstruction).item()}')
                print(f'MSE Segmentation: {loss_segmentation.item()}')
                print(f'PSNR Segmentation: {psnr_loss(approximated_segmentation, mask).item()}')

                writer.add_scalar('Reconstruction Loss', loss_recontruction.item(), global_step=index+epoch*train_dataloader.__len__())
                writer.add_scalar('PSNR Loss', psnr_loss(approximated_reconstruction, reconstruction).item(), global_step=index+epoch*train_dataloader.__len__())
                writer.add_scalar('Reconstruction Loss', loss_recontruction.item(), global_step=index+epoch*train_dataloader.__len__())
                writer.add_scalar('Segmentation Loss', loss_segmentation.item(), global_step=index+epoch*train_dataloader.__len__())
                writer.add_scalar('Total Loss', loss_total.item(), global_step=index+epoch*train_dataloader.__len__())


            if index %100 == 0:
                writer.add_image('Reconstruction', approximated_reconstruction[0,0].detach().cpu(), dataformats='HW')
                writer.add_image('Segmentation', approximated_segmentation[0,0].detach().cpu(), dataformats='HW')
                image_writer.write_image_tensor(approximated_reconstruction, 'current_reconstruction.jpg')
                image_writer.write_image_tensor(approximated_segmentation, 'current_segmentation.jpg')

            if (index !=0 and index %1000 == 0):
                reconstruction_model_save_path = pathlib.Path(architecture_dict['reconstruction']['save_path'])
                reconstruction_model_save_path.parent.mkdir(exist_ok=True, parents=True)
                torch.save(reconstruction_net.state_dict(), reconstruction_model_save_path)
                segmentation_model_save_path = pathlib.Path(architecture_dict['segmentation']['save_path'])
                segmentation_model_save_path.parent.mkdir(exist_ok=True, parents=True)
                torch.save(segmentation_net.state_dict(), segmentation_model_save_path)



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

        writer.add_scalar('Reconstruction Loss on Test Set', test_loss_reconstruction, global_step=epoch)
        writer.add_scalar('Segmentation Loss on Test Set', test_loss_segmentation, global_step=epoch)
        writer.add_image(f'Reconstruction at step {epoch}', approximated_reconstruction[0,0], dataformats='HW', global_step=epoch) #type:ignore
        writer.add_image(f'Segmentation at step {epoch}', approximated_segmentation[0,0], dataformats='HW', global_step=epoch) #type:ignore

        if test_loss_total <= max_tot_loss:
            reconstruction_model_save_path = pathlib.Path(architecture_dict['reconstruction']['save_path'])
            reconstruction_model_save_path.parent.mkdir(exist_ok=True, parents=True)
            segmentation_model_save_path = pathlib.Path(architecture_dict['segmentation']['save_path'])
            segmentation_model_save_path.parent.mkdir(exist_ok=True, parents=True)

            torch.save(reconstruction_net.state_dict(), reconstruction_model_save_path)
            torch.save(segmentation_net.state_dict(), segmentation_model_save_path)

        reconstruction_net.train()
        segmentation_net.train()

    print('Training Finished \u2713 ')

def train_reconstruction_network(
        dimension:int,
        odl_backend:ODLBackend,
        architecture_dict:Dict,
        training_dict:Dict,
        train_dataloader:DataLoader,
        image_writer:PyPlotImageWriter,
        run_writer:SummaryWriter,
        save_folder_path:pathlib.Path,
        verbose=True
        ):

    reconstruction_device = torch.device(architecture_dict['reconstruction']['device_name'])

    reconstruction_net = LearnedPrimalDual(
        dimension = dimension,
        odl_backend = odl_backend,
        n_primal=architecture_dict['reconstruction']['n_primal'],
        n_dual=architecture_dict['reconstruction']['n_dual'],
        n_iterations = architecture_dict['reconstruction']['lpd_n_iterations'],
        n_filters_primal = architecture_dict['reconstruction']['lpd_n_filters_primal'],
        n_filters_dual = architecture_dict['reconstruction']['lpd_n_filters_dual'],
        fourier_filtering = architecture_dict['reconstruction']['fourier_filtering'],
        device = reconstruction_device
        )

    reconstruction_net = load_network(save_folder_path, reconstruction_net, architecture_dict['reconstruction']['load_path'])

    # reconstruction_net = torch.compile(reconstruction_net)

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
    reconstruction_model_file_save_path = reconstruction_model_save_path.joinpath(architecture_dict['reconstruction']['save_path'])

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

    print('Training Finished \u2713 ')

def train_segmentation_network(
        architecture_dict:Dict,
        training_dict:Dict,
        train_dataloader:DataLoader,
        test_dataloader:DataLoader,
        image_writer:PyPlotImageWriter
        ):

    segmentation_device = torch.device(architecture_dict['segmentation']['device_name'])

    segmentation_net = Unet(
        dimension=2,
        n_channels_input  = architecture_dict['segmentation']['Unet_input_channels'],
        n_channels_output = architecture_dict['segmentation']['Unet_output_channels'],
        n_filters = architecture_dict['segmentation']['Unet_n_filters']
        ).to(segmentation_device)

    segmentation_net = load_network(segmentation_net, architecture_dict['segmentation']['load_path'])

    segmentation_loss = torch.nn.BCELoss()
    psnr_loss = PSNR()

    optimiser = torch.optim.Adam(
        params = segmentation_net.parameters(),
        lr = training_dict['learning_rate'],
        betas=(0.9,0.99)
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer = optimiser,
        T_max = training_dict['n_epochs']
    )

    pathlib.Path(f'/local/scratch/public/ev373/runs/segmentation').mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(
        log_dir = f'/local/scratch/public/ev373/runs/segmentation'
    )

    max_seg_loss = 1e8
    pathlib.Path('images/segmentation').mkdir(parents=True, exist_ok=True)

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

            if index %10 == 0:
                print(f'Metrics at step {index} of epoch {epoch}')
                print(f'MSE : {loss_segmentation.item()}')
                print(f'PSNR : {psnr_loss(loss_segmentation, reconstruction).item()}')
                writer.add_scalar('Reconstruction Loss', loss_segmentation.item(), global_step=index+epoch*train_dataloader.__len__())
                writer.add_scalar('PSNR Loss', psnr_loss(loss_segmentation, reconstruction).item(), global_step=index+epoch*train_dataloader.__len__())

            if index %100 == 0:
                writer.add_image('Segmentation', approximated_segmentation[0,0].detach().cpu(), dataformats='HW')
                image_writer.write_image_tensor(approximated_segmentation, 'current_segmentation.jpg')

            if (index !=0 and index %1000 == 0):
                segmentation_model_save_path = pathlib.Path(architecture_dict['segmentation']['save_path'])
                segmentation_model_save_path.parent.mkdir(exist_ok=True, parents=True)
                torch.save(segmentation_net.state_dict(), segmentation_model_save_path)

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

        writer.add_scalar('Segmentation Loss on Test Set', test_loss_segmentation, global_step=epoch)
        writer.add_image(f'Segmentation at step {epoch}', approximated_segmentation[0,0], dataformats='HW', global_step=epoch)#type:ignore

        if test_loss_segmentation <= max_seg_loss:
            segmentation_model_save_path = pathlib.Path(architecture_dict['segmentation']['save_path'])
            segmentation_model_save_path.parent.mkdir(exist_ok=True, parents=True)
            torch.save(segmentation_net.state_dict(), segmentation_model_save_path)

        segmentation_net.train()

    print('Training Finished \u2713 ')
