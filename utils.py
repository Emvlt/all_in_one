from typing import Dict, List
import pathlib

import torch
import matplotlib.pyplot as plt

class PyPlotImageWriter():
    def __init__(self, path_to_images_folder: pathlib.Path) -> None:
        self.path_to_images_folder = path_to_images_folder
        self.path_to_images_folder.mkdir(parents=True, exist_ok=True)

    def write_image_tensor(self, x:torch.Tensor, image_name:str):
        plt.matshow(x[0,0].detach().cpu())
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

def check_reconstruction_network_consistency(reconstruction_dict:Dict, training_dict:Dict):
    assert 'device_name' in reconstruction_dict.keys()

    if reconstruction_dict['name']  == 'lpd':
        assert 'dual_loss_weighting' in training_dict.keys(), 'Provide dual_loss_weighting argument to training_dict'
        assert 0<=training_dict['dual_loss_weighting'] <= 1, f"Dual loss must be in ]0,1], currently is {training_dict['dual_loss_weighting']}"

        assert "reconstruction_loss" in training_dict.keys(), 'Provide reconstruction argument to the training_dict'
        assert "sinogram_loss" in training_dict.keys(), 'Provide reconstruction argument to the training_dict'

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

        assert "batch_size" in training_dict.keys(), "provide batch size"
        check_integer('batch_size', training_dict['batch_size'])

        assert "learning_rate" in training_dict.keys(), "provide learning rate"
        check_float('learning_rate', training_dict['learning_rate'])

        assert "n_epochs" in training_dict.keys(), "provide number of training epochs"
        check_integer('n_epochs', training_dict['n_epochs'])

        if reconstruction_dict['fourier_filtering']:
            assert 'filter_name' in reconstruction_dict.keys(), 'Provide filter_name str argument'
            check_string('filter_name', reconstruction_dict['filter_name'])

            assert 'train_filter' in reconstruction_dict.keys(), 'Provide train_filter boolean argument'
            check_boolean('train_filter', reconstruction_dict['train_filter'])

        assert 'load_path' in reconstruction_dict.keys(), 'specify load path for reconstruction network'
        assert 'save_path' in reconstruction_dict.keys(), 'specify save path for reconstruction network'

    elif reconstruction_dict['name'] =='filtered_backprojection':
        assert 'filter_name' in reconstruction_dict.keys(), 'Provide filter_name str argument'
        check_string('filter_name', reconstruction_dict['filter_name'])

    elif reconstruction_dict['name'] =='backprojection':
        pass

    elif reconstruction_dict['name'] =='fourier_filtering':
        assert 'filter_name' in reconstruction_dict.keys(), 'Provide filter_name str argument'
        check_string('filter_name', reconstruction_dict['filter_name'])

        assert 'train_filter' in reconstruction_dict.keys(), 'Provide train_filter str argument'
        check_boolean('train_filter', reconstruction_dict['train_filter'])

    else:
        raise NotImplementedError(f"Reconstruction network {reconstruction_dict['name']} not implemented.")

def consistency_checks(metadata_dict:Dict):
    assert "pipeline" in metadata_dict.keys(), 'Provide pipeline argument to metadata'
    architecture_dict = metadata_dict['architecture_dict']
    training_dict = metadata_dict['training_dict']

    if metadata_dict["pipeline"] == 'reconstruction':
        assert 'dose' in training_dict.keys(), 'Provide dose argument to training_dict'
        assert 0< training_dict['dose'] <= 1, f"Dose must be in ]0,1], currently is {training_dict['dose']}"

        assert 'reconstruction' in architecture_dict.keys(), 'Specify reconstruction network'
        check_dict('reconstruction', architecture_dict['reconstruction'])

        reconstruction_dict = architecture_dict['reconstruction']
        check_reconstruction_network_consistency(reconstruction_dict, training_dict)

    else:
        raise NotImplementedError (f'Consistency checks not implemented for {metadata_dict["pipeline"]}')

def check_training_dict(training_dict:Dict, pipeline:str):
    assert "training_proportion" in training_dict.keys(), "provide training set proportion"
    check_float('training_proportion', training_dict['training_proportion'])

    assert "num_workers" in training_dict.keys(), "provide number of dataloader workers"
    check_integer('num_workers', training_dict['num_workers'])

    assert "is_subset" in training_dict.keys(), "provide is_subset boolean argument"
    check_boolean('is_subset', training_dict['is_subset'])

    if training_dict['is_subset']:
        assert "subset" in training_dict.keys(), "provide subset list argument"
        check_list('subset', training_dict['subset'])

def check_metadata(metadata_dict:Dict):
    try:
        print('Checking metadata type and parameters consistency...')

        assert 'pipeline' in metadata_dict.keys(), 'Provide pipeline string value: joint, sequential, end_to_end'
        check_string('pipeline', metadata_dict['pipeline'])

        assert 'dimension' in metadata_dict.keys(), 'Provide dimension parameter'
        check_integer('dimension', metadata_dict['dimension'])

        assert 'experiment_folder_name' in metadata_dict.keys(), 'Provide experiment_folder_name string value'
        check_string('experiment_folder_name', metadata_dict['experiment_folder_name'])

        assert 'run_name' in metadata_dict.keys(), 'Provide run_name string value'
        check_string('run_name', metadata_dict['run_name'])

        assert 'training_dict' in metadata_dict.keys(), 'Provide training dictionnary'
        check_training_dict(metadata_dict['training_dict'], metadata_dict['pipeline'])

        assert 'scan_parameter_dict' in metadata_dict.keys(), 'Provide scan parameter dictionnary'
        check_scan_parameter_dict(metadata_dict['scan_parameter_dict'])

        assert 'architecture_dict' in metadata_dict.keys(), 'Provide architecture dictionnary'

        consistency_checks(metadata_dict)

        print("Metadata sanity checks passed \u2713 ")

    except KeyError as error_message:
        return error_message

    except AssertionError as error_message:
        return error_message


