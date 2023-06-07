from typing import Dict
import pathlib

import torch
import matplotlib.pyplot as plt

class PyPlotImageWriter():
    def __init__(self, path_to_images_folder: pathlib.Path) -> None:
        self.path_to_images_folder = path_to_images_folder
        self.path_to_images_folder.mkdir(parents=True, exist_ok=True)

    def write_image_tensor(self, x:torch.Tensor, image_name:str):
        plt.matshow(x[0,0].detach().cpu())
        plt.savefig(self.path_to_images_folder.joinpath(image_name))
        plt.clf()
        plt.close()

def check_integer(variable_name:str, variable_value):
    assert type(variable_value) == int, f'{variable_name} must be an integer'

def check_float(variable_name:str, variable_value):
    assert type(variable_value) == float, f'{variable_name} must be a float'

def check_string(variable_name:str, variable_value):
    assert type(variable_value) == str, f'{variable_name} must be a string'

def check_dict(variable_name:str, variable_value):
    assert type(variable_value) == type({}), f'{variable_name} must be a dictionnary'

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

def check_reconstruction_network_consistency(reconstruction_dict:Dict):
    assert 'load_path' in reconstruction_dict.keys(), 'specify load path for reconstruction network'
    assert 'save_path' in reconstruction_dict.keys(), 'specify save path for reconstruction network'

    if reconstruction_dict['name']  == 'lpd':
       assert 'lpd_n_iterations' in reconstruction_dict.keys(), 'Specify number of lpd iterations'
       check_integer('lpd_n_iterations', reconstruction_dict['lpd_n_iterations'])

       assert 'lpd_n_filters' in reconstruction_dict.keys(), 'Specify number of lpd filters'
       check_integer('lpd_n_filters', reconstruction_dict['lpd_n_filters'])

    else:
        raise NotImplementedError(f"Reconstruction network {reconstruction_dict['name']} not implemented.")

def check_segmentation_network_consistency(segmentation_dict:Dict):
    assert 'load_path' in segmentation_dict.keys(), 'specify load path for reconstruction network'
    assert 'save_path' in segmentation_dict.keys(), 'specify save path for reconstruction network'

    if segmentation_dict['name'] == 'Unet':
        assert 'Unet_input_channels' in  segmentation_dict.keys(), 'Specify number of input channels'
        check_integer('Unet_input_channels', segmentation_dict['Unet_input_channels'])

        assert 'Unet_output_channels' in  segmentation_dict.keys(), 'Specify number of output channels'
        check_integer('Unet_output_channels', segmentation_dict['Unet_output_channels'])

        assert 'Unet_n_filters' in  segmentation_dict.keys(), 'Specify number of filters'
        check_integer('Unet_n_filters', segmentation_dict['Unet_n_filters'])

def check_end_to_end_consistency(architecture_dict:Dict):
    ### Does the key exist?
    assert 'reconstruction' in architecture_dict.keys(), 'Specify reconstruction network'
    assert 'segmentation' in architecture_dict.keys(), 'Specify segmentation network'

    ### Is the value a dictionary?
    check_dict('reconstruction', architecture_dict['reconstruction'])
    check_dict('segmentation', architecture_dict['segmentation'])

    reconstruction_dict = architecture_dict['reconstruction']
    segmentation_dict = architecture_dict['segmentation']

    ### Are the networks consistent?
    check_reconstruction_network_consistency(reconstruction_dict)
    check_segmentation_network_consistency(segmentation_dict)

    ### Is the save path coherent?
    assert pathlib.Path(reconstruction_dict["save_path"]).parent.stem == 'end_to_end', f'Wrong save folder path, specify parent folder as end_to_end, not {pathlib.Path(reconstruction_dict["save_path"]).parent.stem}'
    assert pathlib.Path(segmentation_dict["save_path"]).parent.stem == 'end_to_end', f'Wrong save folder path, specify parent folder as end_to_end, not {pathlib.Path(segmentation_dict["save_path"]).parent.stem}'

def check_joint_consistency(architecture_dict:Dict):
    ### Does the key exist?
    assert 'reconstruction' in architecture_dict.keys(), 'Specify reconstruction network'
    assert 'segmentation' in architecture_dict.keys(), 'Specify segmentation network'

    ### Is the value a dictionary?
    check_dict('reconstruction', architecture_dict['reconstruction'])
    check_dict('segmentation', architecture_dict['segmentation'])

    reconstruction_dict = architecture_dict['reconstruction']
    segmentation_dict = architecture_dict['segmentation']

    ### Are the networks consistent?
    check_reconstruction_network_consistency(reconstruction_dict)
    check_segmentation_network_consistency(segmentation_dict)

    ### Is the save path coherent?
    assert pathlib.Path(reconstruction_dict["save_path"]).parent.stem == 'joint', f'Wrong save folder path, specify parent folder as joint, not {pathlib.Path(reconstruction_dict["save_path"]).parent.stem}'
    assert pathlib.Path(segmentation_dict["save_path"]).parent.stem == 'joint', f'Wrong save folder path, specify parent folder as joint, not {pathlib.Path(segmentation_dict["save_path"]).parent.stem}'

def check_reconstruction_consistency(architecture_dict:Dict):
    ### Does the key exist?
    assert 'reconstruction' in architecture_dict.keys(), 'Specify reconstruction network'

    ### Is the value a dictionary?
    check_dict('reconstruction', architecture_dict['reconstruction'])

    reconstruction_dict = architecture_dict['reconstruction']

    ### Are the networks consistent?
    check_reconstruction_network_consistency(reconstruction_dict)

    ### Is the save path coherent?
    assert pathlib.Path(reconstruction_dict["save_path"]).parent.stem == 'reconstruction', f'Wrong save folder path, specify parent folder as reconstruction, not {pathlib.Path(reconstruction_dict["save_path"]).parent.stem}'

def check_segmentation_consistency(architecture_dict:Dict):
    ### Does the key exist?
    assert 'segmentation' in architecture_dict.keys(), 'Specify segmentation network'

    ### Is the value a dictionary?
    check_dict('segmentation', architecture_dict['segmentation'])

    segmentation_dict = architecture_dict['segmentation']

    ### Are the networks consistent?
    check_segmentation_network_consistency(segmentation_dict)

    ### Is the save path coherent?
    assert pathlib.Path(segmentation_dict["save_path"]).parent.stem == 'segmentation', f'Wrong save folder path, specify parent folder as segmentation, not {pathlib.Path(segmentation_dict["save_path"]).parent.stem}'

def consistency_checks(metada_dict:Dict):
    assert "pipeline" in metada_dict.keys(), 'Provide pipeline argument to metadata'

    if metada_dict["pipeline"] == 'joint':
        check_joint_consistency(metada_dict['architecture_dict'])

        assert "C" in metada_dict['training_dict'].keys(), "provide C parameter value"
        assert 0< metada_dict['training_dict']['C'] <= 1, f"C must be in [0,1], currently is {metada_dict['training_dict']['C']}"

        assert 'dose' in metada_dict['training_dict'].keys(), 'Provide dose argument to metadata'
        assert 0< metada_dict['training_dict']['dose'] <= 1, f"Dose must be in ]0,1], currently is {metada_dict['training_dict']['dose']}"

    elif metada_dict["pipeline"] == 'reconstruction':
        check_reconstruction_consistency(metada_dict['architecture_dict'])

        assert 'dose' in metada_dict['training_dict'].keys(), 'Provide dose argument to metadata'
        assert 0< metada_dict['training_dict']['dose'] <= 1, f"Dose must be in ]0,1], currently is {metada_dict['training_dict']['dose']}"

    elif metada_dict["pipeline"] == 'segmentation':
        check_segmentation_consistency(metada_dict['architecture_dict'])

    elif metada_dict["pipeline"] == 'end_to_end':
        check_end_to_end_consistency(metada_dict['architecture_dict'])

        assert 'dose' in metada_dict['training_dict'].keys(), 'Provide dose argument to metadata'
        assert 0< metada_dict['training_dict']['dose'] <= 1, f"Dose must be in ]0,1], currently is {metada_dict['training_dict']['dose']}"


    else:
        raise NotImplementedError (f'Consistency checks not implemented for {metada_dict["pipeline"]}')

def check_training_dict(training_dict:Dict):
    assert "batch_size" in training_dict.keys(), "provide batch size"
    check_integer('batch_size', training_dict['batch_size'])

    assert "training_proportion" in training_dict.keys(), "provide training set proportion"
    check_float('training_proportion', training_dict['training_proportion'])

    assert "learning_rate" in training_dict.keys(), "provide learning rate"
    check_float('learning_rate', training_dict['learning_rate'])

    assert "n_epochs" in training_dict.keys(), "provide number of training epochs"
    check_integer('n_epochs', training_dict['n_epochs'])

    assert "num_workers" in training_dict.keys(), "provide number of dataloader workers"
    check_integer('num_workers', training_dict['num_workers'])

def check_metadata(metada_dict:Dict):
    print('Checking metadata type and parameters consistency...')
    assert 'dimension' in metada_dict.keys(), 'Provide dimension parameter'
    check_integer('dimension', metada_dict['dimension'])

    assert 'scan_parameter_dict' in metada_dict.keys(), 'Provide scan parameter dictionnary'
    check_scan_parameter_dict(metada_dict['scan_parameter_dict'])

    assert 'pipeline' in metada_dict.keys(), 'Provide pipeline string value: joint, sequential, end_to_end'

    assert 'training_dict' in metada_dict.keys(), 'Provide training dictionnary'
    check_training_dict(metada_dict['training_dict'])

    assert 'architecture_dict' in metada_dict.keys(), 'Provide architecture dictionnary'

    consistency_checks(metada_dict)

    print("Metadata sanity checks passed \u2713 ")

