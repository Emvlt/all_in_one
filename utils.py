from typing import Dict, List
import pathlib

import json
import torch
import matplotlib.pyplot as plt

def load_json(file_path:pathlib.Path):
    if not file_path.is_file():
        raise FileNotFoundError (f'No file found at {file_path}')
    with open(file_path, 'r') as file_read:
        file = json.load(file_read)
    return file

def save_json(file_path:pathlib.Path | str, file:Dict):
    with open(file_path, 'w') as file_write:
        json.dump(file, file_write)

class PyPlotImageWriter():
    def __init__(self, path_to_images_folder: pathlib.Path) -> None:
        self.path_to_images_folder = path_to_images_folder
        self.path_to_images_folder.mkdir(parents=True, exist_ok=True)

    def write_image_tensor(self, x:torch.Tensor, image_name:str):
        x = x.detach().cpu()
        ## unholy
        while len(x.size()) != 2:
            x = x[0]
        plt.matshow(x)
        plt.axis('off')
        plt.savefig(self.path_to_images_folder.joinpath(image_name), bbox_inches='tight')
        plt.clf()
        plt.close()

    def write_line_tensor(self, x:torch.Tensor, image_name:str):
        plt.plot(x.detach().cpu())
        plt.savefig(self.path_to_images_folder.joinpath(image_name))
        plt.clf()
        plt.close()

    def write_kernel_weights(self, x:List[torch.Tensor], names:List[str], fig_name:str):
        f, (axs) = plt.subplots(len(names), sharey=True)
        for i, weight in enumerate(x):
            axs[i].plot(weight.detach().cpu())
            axs[i].set_title(names[i])
        plt.savefig(self.path_to_images_folder.joinpath(fig_name))
        plt.clf()
        plt.close()

def check_boolean(variable_name:str, variable_value):
    assert type(variable_value) == bool, f'{variable_name} must be a boolean'

def check_integer(variable_name:str, variable_value):
    assert type(variable_value) == int, f'{variable_name} must be an integer'

def check_float(variable_name:str, variable_value):
    assert type(variable_value) == float, f'{variable_name} must be a float'

def check_string(variable_name:str, variable_value):
    assert type(variable_value) == str, f'{variable_name} must be a string'

def check_dict(variable_name:str, variable_value):
    assert type(variable_value) == type({}), f'{variable_name} must be a dictionnary'

def check_list(variable_name:str, variable_value):
    assert type(variable_value) == type([]), f'{variable_name} must be a list'

def check_scan_parameter_dict(scan_parameter_dict:Dict):
    assert 'space_dict' in scan_parameter_dict.keys(), 'Provide a space dictionnary'
    check_dict('space_dict', scan_parameter_dict['space_dict'])
    assert 'min_pt' in scan_parameter_dict['space_dict'].keys(), 'min_pt not provided in space dictionary'
    assert 'max_pt' in scan_parameter_dict['space_dict'].keys(), 'max_pt not provided in space dictionary'
    assert 'shape' in scan_parameter_dict['space_dict'].keys(), 'shape not provided in space dictionary'
    assert 'dtype' in scan_parameter_dict['space_dict'].keys(), 'dtype not provided in space dictionary'

    assert 'angle_partition_dict' in scan_parameter_dict.keys(), 'Provide a angle partition dictionnary'
    check_dict('angle_partition_dict', scan_parameter_dict['angle_partition_dict'])
    assert 'min_pt' in scan_parameter_dict['angle_partition_dict'].keys(), 'min_pt not provided in angle partition dictionary'
    assert 'max_pt' in scan_parameter_dict['angle_partition_dict'].keys(), 'max_pt not provided in angle partition dictionary'
    assert 'shape' in scan_parameter_dict['angle_partition_dict'].keys(), 'shape not provided in angle partition dictionary'

    assert 'detector_partition_dict' in scan_parameter_dict.keys(), 'Provide a detector partition dictionnary'
    check_dict('detector_partition_dict', scan_parameter_dict['detector_partition_dict'])
    assert 'min_pt' in scan_parameter_dict['detector_partition_dict'].keys(), 'min_pt not provided in detector partition dictionary'
    assert 'max_pt' in scan_parameter_dict['detector_partition_dict'].keys(), 'max_pt not provided in detector partition dictionary'
    assert 'shape' in scan_parameter_dict['detector_partition_dict'].keys(), 'shape not provided in detector partition dictionary'
    #assert 'cell_sides' in scan_parameter_dict['detector_partition_dict'].keys(), 'cell_sides not provided in detector partition dictionary'

    assert 'geometry_dict' in scan_parameter_dict.keys(), 'Provide a geometry dictionnary'
    check_dict('geometry_dict', scan_parameter_dict['geometry_dict'])
    assert 'src_radius' in scan_parameter_dict['geometry_dict'], 'src_radius not provided in geometry dictionary'
    assert 'det_radius' in scan_parameter_dict['geometry_dict'], 'det_radius not provided in geometry dictionary'
    assert 'beam_geometry' in scan_parameter_dict['geometry_dict'], 'beam_geometry not provided in geometry dictionary'

def check_training_dict(training_dict:Dict):
    assert "num_workers" in training_dict.keys(), "provide number of dataloader workers"
    check_integer('num_workers', training_dict['num_workers'])

    assert "batch_size" in training_dict.keys(), "provide batch size"
    check_integer('batch_size', training_dict['batch_size'])

    assert "learning_rate" in training_dict.keys(), "provide learning rate"
    check_float('learning_rate', training_dict['learning_rate'])

    assert "n_epochs" in training_dict.keys(), "provide number of training epochs"
    check_integer('n_epochs', training_dict['n_epochs'])

def check_reconstruction_network_consistency(reconstruction_dict:Dict, metadata_dict:Dict):
    assert 'device_name' in reconstruction_dict.keys(), 'Specify reconstruction device name'

    assert "train" in reconstruction_dict.keys(), 'Specify training mode'
    check_boolean('train', reconstruction_dict['train'])

    if reconstruction_dict['name']  == 'lpd':
        assert 'dimension' in reconstruction_dict.keys(), 'Specify dimension of LPD architecture'

        if reconstruction_dict['train']:
            assert 'training_dict' in metadata_dict.keys(), 'Provide training dictionnary'
            check_training_dict(metadata_dict['training_dict'])
            training_dict = metadata_dict['training_dict']

            assert 'dual_loss_weighting' in training_dict.keys(), 'Provide dual_loss_weighting argument to training_dict'
            assert 0<=training_dict['dual_loss_weighting'] <= 1, f"Dual loss must be in ]0,1], currently is {training_dict['dual_loss_weighting']}"

            assert "reconstruction_loss" in training_dict.keys(), 'Provide reconstruction argument to the training_dict'
            assert "sinogram_loss" in training_dict.keys(), 'Provide reconstruction argument to the training_dict'

            assert 'save_path' in reconstruction_dict.keys(), 'specify save path for reconstruction network'

        else:
            assert 'load_path' in reconstruction_dict.keys(), 'specify load path for reconstruction network'

        assert 'n_primal' in reconstruction_dict.keys(), 'Specify number of lpd primal channels'
        check_integer('n_primal', reconstruction_dict['n_primal'])

        assert 'n_dual' in reconstruction_dict.keys(), 'Specify number of lpd dual channels'
        check_integer('n_dual', reconstruction_dict['n_dual'])

        assert 'lpd_n_iterations' in reconstruction_dict.keys(), 'Specify number of lpd iterations'
        check_integer('lpd_n_iterations', reconstruction_dict['lpd_n_iterations'])

        assert 'lpd_n_filters_primal' in reconstruction_dict.keys(), 'Specify number of lpd filters primal'
        check_integer('lpd_n_filters_primal', reconstruction_dict['lpd_n_filters_primal'])

        assert 'lpd_n_filters_dual' in reconstruction_dict.keys(), 'Specify number of lpd filters dual'
        check_integer('lpd_n_filters_dual', reconstruction_dict['lpd_n_filters_dual'])

        assert 'fourier_filtering' in reconstruction_dict.keys()
        check_boolean('fourier_filtering', reconstruction_dict['fourier_filtering'])

        if reconstruction_dict['fourier_filtering']:
            assert 'filter_name' in reconstruction_dict.keys(), 'Provide filter_name str argument'
            check_string('filter_name', reconstruction_dict['filter_name'])

            assert 'train_filter' in reconstruction_dict.keys(), 'Provide train_filter boolean argument'
            check_boolean('train_filter', reconstruction_dict['train_filter'])

    elif reconstruction_dict['name'] =='filtered_backprojection':
        assert 'filter_name' in reconstruction_dict.keys(), 'Provide filter_name str argument'
        check_string('filter_name', reconstruction_dict['filter_name'])

    elif reconstruction_dict['name'] =='backprojection':
        pass

    elif reconstruction_dict['name'] =='fourier_filtering':
        assert 'filter_name' in reconstruction_dict.keys(), 'Provide filter_name str argument'
        check_string('filter_name', reconstruction_dict['filter_name'])

        assert 'dimension' in reconstruction_dict.keys(), 'Provide dimension int argument'
        check_integer('dimension', reconstruction_dict['dimension'])

        if reconstruction_dict['train']:
            assert 'training_dict' in metadata_dict.keys(), 'Provide training dictionnary'
            check_training_dict(metadata_dict['training_dict'])

    else:
        raise NotImplementedError(f"Reconstruction network {reconstruction_dict['name']} not implemented.")

def check_segmentation_network_consistency(segmentation_dict:Dict, metadata_dict:Dict):
    assert 'device_name' in segmentation_dict.keys(), 'Specify segmentation device name'

    assert "train" in segmentation_dict.keys(), 'Specify training mode channels'
    check_boolean('train', segmentation_dict['train'])

    if segmentation_dict['name']  == 'Unet':
        if segmentation_dict['train']:
            assert 'training_dict' in metadata_dict.keys(), 'Provide training dictionnary'
            check_training_dict(metadata_dict['training_dict'])
            training_dict = metadata_dict['training_dict']

            assert "segmentation_loss" in training_dict.keys(), 'Provide segmentation_loss argument to the training_dict'

            assert 'save_path' in segmentation_dict.keys(), 'specify save path for reconstruction network'

            assert "reconstructed" in training_dict.keys(), 'Specify reconstructed boolean argument'
            check_boolean('reconstructed', training_dict['reconstructed'])

        else:
            assert 'load_path' in segmentation_dict.keys(), 'specify load path for reconstruction network'


        assert 'Unet_input_channels' in segmentation_dict.keys(), 'Specify number of Unet input channels'
        check_integer('Unet_input_channels', segmentation_dict['Unet_input_channels'])

        assert 'Unet_output_channels' in segmentation_dict.keys(), 'Specify number of Unet output channels'
        check_integer('Unet_output_channels', segmentation_dict['Unet_output_channels'])

        assert 'Unet_n_filters' in segmentation_dict.keys(), 'Specify number of Unet filters'
        check_integer('Unet_n_filters', segmentation_dict['Unet_n_filters'])

    else:
        raise NotImplementedError(f"Segmentation network {segmentation_dict['name']} not implemented.")

def check_architecture_consistency(architecture_name:str, metadata_dict:Dict, verbose=False):
    architecture_dict = metadata_dict['architecture_dict']
    if architecture_name in ['reconstruction', 'segmentation']:

        if architecture_name == 'reconstruction':
            if verbose:print('Checking reconstruction network...')
            check_reconstruction_network_consistency(architecture_dict[architecture_name], metadata_dict)

        elif architecture_name == 'segmentation':
            if verbose:print('Checking segmentation network...')
            check_segmentation_network_consistency(architecture_dict[architecture_name], metadata_dict)

    else:
        raise NotImplementedError (f'Consistency checks not implemented for architecture {architecture_name}')

def check_data_feeding_consistency(data_feeding_dict:Dict):
    assert 'dataset_name' in data_feeding_dict.keys(), 'Provide datase_name string argument to dict'
    check_string('dataset_name', data_feeding_dict['dataset_name'])

    assert 'train' in data_feeding_dict.keys(), 'Provide train boolean argument to dict'
    check_boolean('train', data_feeding_dict['train'])

    assert 'is_subset' in data_feeding_dict.keys(), 'Provide is_subset argument to dict'
    check_boolean('is_subset', data_feeding_dict['is_subset'])

    if data_feeding_dict['is_subset']:
        assert "subset" in data_feeding_dict.keys(), "provide subset list argument"
        check_list('subset', data_feeding_dict['subset'])

    assert 'training_proportion' in data_feeding_dict.keys(), 'Provide training_proportion argument to dict'

def check_metadata(metadata_dict:Dict, verbose = True):
    if verbose:
        print('Checking metadata type and parameters consistency...')

    assert 'data_feeding_dict' in metadata_dict.keys(), 'Provide data feeding dictionary'
    check_data_feeding_consistency(metadata_dict['data_feeding_dict'])

    assert 'architecture_dict' in metadata_dict.keys(), 'Provide architecture dictionnary'

    for architecture_name in metadata_dict['architecture_dict'].keys():
        check_architecture_consistency(architecture_name, metadata_dict, verbose)

    if verbose:
        print("Metadata sanity checks passed \u2713 ")




