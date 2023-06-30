import argparse
import pathlib
from typing import Dict, Callable, List

import torch
import json
from torchvision.transforms import Compose
import numpy as np

from metrics import PSNR #type:ignore
from models import LearnedPrimalDual, load_network, FourierFilteringModule
from datasets import LIDC_IDRI
from backends.odl import ODLBackend
from utils import check_metadata, PyPlotImageWriter
from transforms import Normalise, ToFloat # type:ignore

def get_inference_function(metadata_dict:Dict, dimension:int, pipeline: str, odl_backend:ODLBackend, **kwargs) -> Callable:
    architecture_dict = metadata_dict['architecture_dict']

    if pipeline == 'reconstruction':
        architecture_name = architecture_dict['reconstruction']['name']
        if architecture_name == 'lpd':
            reconstruction_device = torch.device(architecture_dict['reconstruction']['device_name'])
            lpd_network = LearnedPrimalDual(
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
            load_network(lpd_network, architecture_dict['reconstruction']['save_path'])
            lpd_network.eval()
            return lpd_network.forward

        elif architecture_name =='backprojection':
            return odl_backend.get_reconstruction

        elif architecture_name =='filtered_backprojection':
            return odl_backend.get_filtered_backprojection_operator(filter_name = architecture_dict['reconstruction']['filter_name'])

        elif architecture_name =='fourier_filtering':
            reconstruction_device = torch.device(architecture_dict['reconstruction']['device_name'])
            fourier_filtering_module= FourierFilteringModule(
                dimension = dimension,
                n_measurements = odl_backend.angle_partition_dict['shape'],
                detector_size=odl_backend.detector_partition_dict['shape'],
                device = reconstruction_device,
                filter_name = architecture_dict['reconstruction']['filter_name'],
                training_mode = False
            )
            load_network(fourier_filtering_module, architecture_dict['reconstruction']['save_path'])
            fourier_filtering_module.eval()
            return fourier_filtering_module.forward

        else:
            raise ValueError(f'Wrong name argument in architecture dict: {architecture_name} not accepted')

    elif pipeline == 'segmentation':
        raise NotImplementedError

    elif pipeline =='joint':
        raise NotImplementedError

    elif pipeline == 'end-to-end':
        raise NotImplementedError

    else:
        raise ValueError(f'Wrong pipeline argument: {pipeline} not accepted')


def infer_slice(
    inference_function:Callable, patient_slice_path:pathlib.Path,
    dataset:LIDC_IDRI, reconstruction_device:torch.device,
    sinogram_transforms,
    inference_name
                ):
    reconstruction = dataset.get_reconstruction_tensor(patient_slice_path).unsqueeze(0).to(reconstruction_device)
    sinogram = odl_backend.get_sinogram(reconstruction)
    sinogram = sinogram_transforms(sinogram)

    if inference_name == 'backprojection':
        approximated_reconstruction = inference_function(sinogram)

    elif inference_name == 'filtered_backprojection':
        ### Ugly conversions to numpy needed (are they really?) for fbp with odl backend
        sinogram = sinogram[0,0].detach().cpu().numpy()
        approximated_reconstruction = np.asarray(inference_function(sinogram))
        approximated_reconstruction = torch.from_numpy(approximated_reconstruction).unsqueeze(0).unsqueeze(0).to(reconstruction_device)

    elif inference_name == 'lpd':
        if dimension == 1:
            sinogram = torch.squeeze(sinogram, dim=1)
        approximated_reconstruction, _ = inference_function(sinogram)

    elif inference_name == 'fourier_filtering':
        if dimension == 1:
            sinogram = torch.squeeze(sinogram, dim=1)
        ### will be problematic
        filtered_sinogram:torch.Tensor = inference_function(sinogram)
        approximated_reconstruction = odl_backend.get_reconstruction(filtered_sinogram.unsqueeze(0).unsqueeze(0))

    else:
        raise NotImplementedError(f'Inference not implemented for {inference_name}')
    return approximated_reconstruction, reconstruction

def qualitative_evaluation(pipeline:str, run_name:str, inference_function:Callable , dataset:LIDC_IDRI, patient_index:str, slice_index:int, image_writer:PyPlotImageWriter):
    print(f'Evaluating patient {patient_index}')
    error_function = PSNR()
    if pipeline == 'reconstruction':
        inference_name = metadata_dict["architecture_dict"]["reconstruction"]["name"]
        sinogram_transforms = Normalise()
        reconstruction_device = metadata_dict["architecture_dict"]["reconstruction"]["device_name"]
        patient_slice_path = dataset.get_patient_slice_index_path(patient_index, slice_index)
        approximated_reconstruction, reconstruction = infer_slice(
                inference_function, patient_slice_path,
                dataset, reconstruction_device, sinogram_transforms, inference_name
            )
        image_writer.write_image_tensor(approximated_reconstruction, f'patient_{patient_index}_slice_{slice_index}_{run_name}.jpg')
        image_writer.write_image_tensor(reconstruction, f'patient_{patient_index}_slice_{slice_index}_target.jpg')
        print(error_function(approximated_reconstruction, reconstruction).item())

def evaluate_method(pipeline:str, run_name:str, inference_function:Callable , dataset:LIDC_IDRI) -> Dict:
    ### Currently in a disgusting state ###
    print(f'----------------- Evaluating {pipeline} pipeline: run {run_name} -----------------')
    result_dict = {}
    ## Default error function
    error_function = PSNR()

    for patient_id in dataset.testing_patients_list:
        print(f'Evaluating patient {patient_id}')
        result_dict[patient_id] = []

        if pipeline == 'reconstruction':
            inference_name = metadata_dict["architecture_dict"]["reconstruction"]["name"]
            sinogram_transforms = Normalise()
            reconstruction_device = metadata_dict["architecture_dict"]["reconstruction"]["device_name"]
            patient_indices_list = dataset.get_patient_slices_list(patient_id)
            for patient_slice_path in patient_indices_list:
                approximated_reconstruction, reconstruction = infer_slice(
                    inference_function, patient_slice_path,
                    dataset, reconstruction_device, sinogram_transforms, inference_name
                )
                result_dict[patient_id].append(error_function(approximated_reconstruction, reconstruction).item())

        else:
            raise NotImplementedError

    return result_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_path', required=True)
    parser.add_argument('--platform', required=True)
    parser.add_argument('--quantitative', action='store_true')
    parser.add_argument('--qualitative', dest='quantitative', action='store_false')
    parser.set_defaults(quantitative=True)
    args = parser.parse_args()

    print(f'Running code on {args.platform}')
    ## Unpacking paths
    paths_dict = dict(json.load(open('paths_dict.json')))[args.platform]
    MODELS_PATH = pathlib.Path(paths_dict['MODELS_PATH'])
    RUNS_PATH = pathlib.Path(paths_dict['RUNS_PATH'])
    DATASET_PATH = pathlib.Path(paths_dict['DATASET_PATH'])

    ## Unpacking metadata
    metadata_dict = dict(json.load(open(args.metadata_path)))
    pipeline = metadata_dict['pipeline']
    run_name = metadata_dict["run_name"]
    experiment_folder_name = metadata_dict["experiment_folder_name"]
    training_dict = metadata_dict["training_dict"]
    scan_parameter_dict = metadata_dict["scan_parameter_dict"]
    architecture_dict = metadata_dict['architecture_dict']
    dimension = metadata_dict['dimension']

    ## Sanity checks
    check_metadata(metadata_dict)

    ## Instanciate backend
    odl_backend = ODLBackend()
    odl_backend.initialise_odl_backend_from_metadata_dict(scan_parameter_dict)

    ## Transforms
    transforms = {
        "reconstruction_transforms":Compose([ToFloat(), Normalise()]),
        "mask_transforms":Compose([ToFloat()])
    }

    ## Dataset and Dataloader
    testing_lidc_idri_dataset = LIDC_IDRI(
        DATASET_PATH,
        pipeline,
        odl_backend,
        training_dict['training_proportion'],
        'testing',
        training_dict['is_subset'],
        transform = transforms,
        subset = training_dict['subset']
        )

    inference_function = get_inference_function(metadata_dict, dimension, pipeline, odl_backend)

    if args.quantitative:
        evaluation_dict = evaluate_method(pipeline, run_name, inference_function, testing_lidc_idri_dataset)

        with open(f'results/{pipeline}/{experiment_folder_name}/{run_name}.json', 'w') as out_file:
            json.dump(evaluation_dict, out_file, indent=4)
    else:
        image_writer = PyPlotImageWriter(pathlib.Path(f'images/{pipeline}/{experiment_folder_name}/{run_name}'))
        qualitative_evaluation(pipeline, run_name, inference_function, testing_lidc_idri_dataset,'LIDC-IDRI-1002', 50, image_writer)