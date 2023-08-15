from typing import Dict
import pathlib

def check_boolean(variable_name:str, variable_value):
    assert type(variable_value) == bool, f'{variable_name} must be a boolean, currently {variable_value} is {type(variable_value)}'

def check_integer(variable_name:str, variable_value):
    assert type(variable_value) == int, f'{variable_name} must be an integer, currently {variable_value} is {type(variable_value)}'

def check_float(variable_name:str, variable_value):
    assert type(variable_value) == float, f'{variable_name} must be a float, currently {variable_value} is {type(variable_value)}'

def check_string(variable_name:str, variable_value):
    assert type(variable_value) == str, f'{variable_name} must be a string, currently {variable_value} is {type(variable_value)}'

def check_dict(variable_name:str, variable_value):
    assert type(variable_value) == type({}), f'{variable_name} must be a dictionnary, currently {variable_value} is {type(variable_value)}'

def check_list(variable_name:str, variable_value):
    assert type(variable_value) == type([]), f'{variable_name} must be a list, currently {variable_value} is {type(variable_value)}'

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
    '''assert "learning_rate" in training_dict.keys(), "provide learning rate"
    check_float('learning_rate', training_dict['learning_rate'])'''

    '''assert "n_epochs" in training_dict.keys(), "provide number of training epochs"
    check_integer('n_epochs', training_dict['n_epochs'])'''
    pass

def check_fourier_filtering_network_consistency(fourier_filtering_module_dict:Dict):
    assert 'name' in fourier_filtering_module_dict.keys(), 'Provide name str argument'
    check_string('name',fourier_filtering_module_dict['name'])

    assert 'train' in fourier_filtering_module_dict.keys(), 'Provide train boolean argument'
    check_boolean('train',fourier_filtering_module_dict['train'])

    assert 'dimension' in fourier_filtering_module_dict.keys(), 'Provide dimension int argument'
    check_integer('dimension',fourier_filtering_module_dict['dimension'])


def check_reconstruction_network_consistency(reconstruction_dict:Dict, metadata_dict:Dict):

    if reconstruction_dict['name']  == 'lpd':
        assert 'primal_dict' in reconstruction_dict.keys(), 'Specify LPD architecture primal_dict'
        check_dict('primal_dict', reconstruction_dict['primal_dict'])
        primal_dict:Dict = reconstruction_dict['primal_dict']
        assert 'dimension' in primal_dict.keys(), 'Specify LPD primal dimension'
        check_integer('dimension', primal_dict['dimension'])
        assert 'name' in primal_dict.keys(), 'Specify LPD primal name'
        check_string('name', primal_dict['name'])
        assert 'n_layers' in primal_dict.keys(), 'Specify LPD primal n_layers'
        check_integer('n_layers', primal_dict['n_layers'])
        assert 'n_filters' in primal_dict.keys(), 'Specify LPD primal n_filters'
        check_integer('n_filters',primal_dict ['n_filters'])

        assert 'dual_dict' in reconstruction_dict.keys(), 'Specify LPD architecture dual_dict'
        check_dict('dual_dict', reconstruction_dict['dual_dict'])
        dual_dict:Dict = reconstruction_dict['dual_dict']
        assert 'dimension' in dual_dict.keys(), 'Specify LPD dual dimension'
        check_integer('dimension', dual_dict['dimension'])
        assert 'name' in dual_dict.keys(), 'Specify LPD dual name'
        check_string('name', dual_dict['name'])
        assert 'n_layers' in dual_dict.keys(), 'Specify LPD dual n_layers'
        check_integer('n_layers', dual_dict['n_layers'])
        assert 'n_filters' in dual_dict.keys(), 'Specify LPD dual n_filters'
        check_integer('n_filters',dual_dict ['n_filters'])

        assert 'device_name' in reconstruction_dict.keys(), 'Specify reconstruction device name'

        assert "train" in reconstruction_dict.keys(), 'Specify training mode'
        check_boolean('train', reconstruction_dict['train'])

        if reconstruction_dict['train']:
            assert 'training_dict' in metadata_dict.keys(), 'Provide training dictionnary'
            check_training_dict(metadata_dict['training_dict'])
            training_dict = metadata_dict['training_dict']

            '''
            assert 'dual_loss_weighting' in training_dict.keys(), 'Provide dual_loss_weighting argument to training_dict'
            assert 0<=training_dict['dual_loss_weighting'] <= 1, f"Dual loss must be in ]0,1], currently is {training_dict['dual_loss_weighting']}"
            '''

            assert "reconstruction_loss" in training_dict.keys(), 'Provide reconstruction argument to the training_dict'
            #assert "sinogram_loss" in training_dict.keys(), 'Provide reconstruction argument to the training_dict'

        else:
            assert 'load_path' in reconstruction_dict.keys(), 'specify load path for reconstruction network'

        if 'fourier_filtering_dict' in reconstruction_dict.keys():
            assert 'module_dict' in reconstruction_dict.keys(), 'Provide module dict argument'
            check_dict('module_dict', reconstruction_dict['module_dict'])
            check_fourier_filtering_network_consistency(reconstruction_dict['module_dict'])

    elif reconstruction_dict['name'] =='filtered_backprojection':
        assert 'filter_name' in reconstruction_dict.keys(), 'Provide filter_name str argument'
        check_string('filter_name', reconstruction_dict['filter_name'])

    elif reconstruction_dict['name'] =='backprojection':
        pass

    elif reconstruction_dict['name'] =='fourier_filtering':
        assert 'module_dict' in reconstruction_dict.keys(), 'Provide module dict argument'
        check_dict('module_dict', reconstruction_dict['module_dict'])
        check_fourier_filtering_network_consistency(reconstruction_dict['module_dict'])

    else:
        raise NotImplementedError(f"Reconstruction network {reconstruction_dict['name']} not implemented.")

def check_segmentation_network_consistency(segmentation_dict:Dict, metadata_dict:Dict):
    assert 'device_name' in segmentation_dict.keys(), 'Specify segmentation device name'

    assert "train" in segmentation_dict.keys(), 'Specify training mode channels'
    check_boolean('train', segmentation_dict['train'])


    if segmentation_dict['train']:
            assert 'training_dict' in metadata_dict.keys(), 'Provide training dictionnary'
            check_training_dict(metadata_dict['training_dict'])
            training_dict:Dict = metadata_dict['training_dict']

            assert "segmentation_loss" in training_dict.keys(), 'Provide segmentation_loss argument to the training_dict'

            data_feeding_dict = metadata_dict['data_feeding_dict']

            if 'reconstruction' in metadata_dict['architecture_dict']:
                check_reconstruction_network_consistency(metadata_dict['architecture_dict']['reconstruction'], metadata_dict)


    if segmentation_dict['name']  == 'Unet':
        unet_dict = segmentation_dict['unet_dict']
        assert 'input_channels' in unet_dict.keys(), 'Specify number of Unet input channels'
        check_integer('input_channels', unet_dict['input_channels'])

        assert 'output_channels' in unet_dict.keys(), 'Specify number of Unet output channels'
        check_integer('output_channels', unet_dict['output_channels'])

        assert 'n_filters' in unet_dict.keys(), 'Specify number of Unet filters'
        check_integer('n_filters', unet_dict['n_filters'])

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


    assert 'shuffle' in data_feeding_dict.keys(), 'Provide shuffle argument to dict'
    check_boolean('shuffle', data_feeding_dict['shuffle'])

    assert "num_workers" in data_feeding_dict.keys(), "provide number of dataloader workers"
    check_integer('num_workers', data_feeding_dict['num_workers'])

    assert "batch_size" in data_feeding_dict.keys(), "provide batch size"
    check_integer('batch_size', data_feeding_dict['batch_size'])

    assert 'training_proportion' in data_feeding_dict.keys(), 'Provide training_proportion argument to dict'

def check_file_naming_consistency(scan_parameter_dict:Dict, file_path:pathlib.Path):
    experiment_folder_name = file_path.parent.stem
    folder_name_to_n_measurements ={
        '6_percent_measurements':64,
        '25_percent_measurements':256,
        '100_percent_measurements':1024
    }
    assert folder_name_to_n_measurements[experiment_folder_name] == scan_parameter_dict['angle_partition_dict']['shape']


def check_metadata(metadata_dict:Dict, file_path:pathlib.Path, verbose = True):
    if verbose:
        print('Checking metadata type and parameters consistency...')

    assert 'data_feeding_dict' in metadata_dict.keys(), 'Provide data feeding dictionary'
    check_data_feeding_consistency(metadata_dict['data_feeding_dict'])

    assert 'architecture_dict' in metadata_dict.keys(), 'Provide architecture dictionnary'

    for architecture_name in metadata_dict['architecture_dict'].keys():
        check_architecture_consistency(architecture_name, metadata_dict, verbose)

    if 'scan_parameter_dict' in metadata_dict.keys():
        scan_parameter_dict = metadata_dict['scan_parameter_dict']
        check_scan_parameter_dict(scan_parameter_dict)
        check_file_naming_consistency(scan_parameter_dict, file_path)

    if verbose:
        print("Metadata sanity checks passed \u2713 ")





