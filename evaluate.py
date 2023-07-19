import argparse
import pathlib
from typing import Dict, Callable, List

import torch
import json
from torchvision.transforms import Compose
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from metrics import PSNR  # type:ignore
from models import LearnedPrimalDual, load_network, FourierFilteringModule
from datasets import LIDC_IDRI
from backends.odl import ODLBackend
from utils import PyPlotImageWriter
from metadata_checker import check_metadata
from transforms import Normalise, ToFloat  # type:ignore


def get_inference_function(metadata_dict: Dict, pipeline: str,odl_backend: ODLBackend,experiment_models_folder_path: pathlib.Path) -> Callable:
    architecture_dict = metadata_dict["architecture_dict"]
    if pipeline == "reconstruction":
        architecture_name = architecture_dict["reconstruction"]["name"]
        print(f"\t \t  Loading inference function for {architecture_name}")
        reconstruction_device = torch.device(architecture_dict["reconstruction"]["device_name"])
        if architecture_name == "lpd":
            lpd_network = LearnedPrimalDual(
                odl_backend=odl_backend,
                network_dict=architecture_dict["reconstruction"],
                device=reconstruction_device,
            )
            try:
                load_network(
                    experiment_models_folder_path,
                    lpd_network,
                    architecture_dict["reconstruction"]["save_path"],
                    indent_level=3,
                )
            except KeyError:
                print("No save_path found, loading default model")
            lpd_network.eval()
            return lpd_network.forward

        elif architecture_name == "backprojection":
            return odl_backend.get_reconstruction

        elif architecture_name == "filtered_backprojection":
            return odl_backend.get_filtered_backprojection_operator(
                filter_name=architecture_dict["reconstruction"]["filter_name"]
            )

        elif architecture_name == "fourier_filtering":

            fourier_filtering_module = FourierFilteringModule(
                fourier_filtering_dict=architecture_dict["reconstruction"],
                n_measurements=odl_backend.angle_partition_dict["shape"],
                detector_size=odl_backend.detector_partition_dict["shape"],
                device=reconstruction_device
            )
            try:
                load_network(
                    experiment_models_folder_path,
                    fourier_filtering_module,
                    architecture_dict["reconstruction"]["save_path"],
                    indent_level=3,
                )
            except KeyError:
                print("No save_path found, loading default model")
            fourier_filtering_module.eval()
            return fourier_filtering_module.forward

        else:
            raise ValueError(
                f"Wrong name argument in architecture dict: {architecture_name} not accepted"
            )

    elif pipeline == "segmentation":
        raise NotImplementedError

    elif pipeline == "joint":
        raise NotImplementedError

    elif pipeline == "end-to-end":
        raise NotImplementedError

    else:
        raise ValueError(f"Wrong pipeline argument: {pipeline} not accepted")


def infer_slice(
    odl_backend: ODLBackend,
    inference_function: Callable,
    patient_slice_path: pathlib.Path,
    dataset: LIDC_IDRI,
    reconstruction_device: torch.device,
    sinogram_transforms,
    inference_name,
    reconstruction_transforms: Dict,
):
    reconstruction = (dataset.get_reconstruction_tensor(patient_slice_path)
        .unsqueeze(0)
        .to(reconstruction_device)
    )
    reconstruction = reconstruction_transforms["reconstruction_transforms"](reconstruction)
    sinogram = odl_backend.get_sinogram(reconstruction)
    sinogram = sinogram_transforms(sinogram)

    if inference_name == "backprojection":
        approximated_reconstruction = inference_function(sinogram)

    elif inference_name == "filtered_backprojection":
        ### Ugly conversions to numpy needed (are they really?) for fbp with odl backend
        sinogram = sinogram[0, 0].detach().cpu().numpy()
        approximated_reconstruction = np.asarray(inference_function(sinogram))
        approximated_reconstruction = (
            torch.from_numpy(approximated_reconstruction)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(reconstruction_device)
        )

    elif inference_name == "lpd":
            approximated_reconstruction, _ = inference_function(sinogram)

    elif inference_name == "fourier_filtering":
        filtered_sinogram: torch.Tensor = inference_function(sinogram)
        approximated_reconstruction = odl_backend.get_reconstruction(
            filtered_sinogram.unsqueeze(0)
        )

    else:
        raise NotImplementedError(f"Inference not implemented for {inference_name}")
    return approximated_reconstruction, reconstruction

def qualitative_evaluation(
    metadata_dict: Dict,
    pipeline: str,
    odl_backend: ODLBackend,
    run_name: str,
    inference_function: Callable,
    dataset: LIDC_IDRI,
    patient_index: str,
    slice_index: int,
    image_writer: PyPlotImageWriter,
    transforms,
):
    print(
        f"\t \t  ------- Qualitative evaluation of {pipeline} pipeline: run {run_name} -------"
    )
    print(f"\t \t \t  Evaluating patient {patient_index}")
    error_function = PSNR()
    if pipeline == "reconstruction":
        inference_name = metadata_dict["architecture_dict"]["reconstruction"]["name"]
        sinogram_transforms = Normalise()
        reconstruction_device = metadata_dict["architecture_dict"]["reconstruction"][
            "device_name"
        ]
        patient_slice_path = dataset.get_patient_slice_index_path(
            patient_index, slice_index
        )
        approximated_reconstruction, reconstruction = infer_slice(
            odl_backend,
            inference_function,
            patient_slice_path,
            dataset,
            reconstruction_device,
            sinogram_transforms,
            inference_name,
            transforms,
        )
        image_writer.write_image_tensor(
            approximated_reconstruction,
            f"patient_{patient_index}_slice_{slice_index}_{run_name}.jpg",
        )
        image_writer.write_image_tensor(
            reconstruction, f"patient_{patient_index}_slice_{slice_index}_target.jpg"
        )
        print(
            f"\t \t  \t {error_function(approximated_reconstruction, reconstruction).item()}"
        )
    else:
        raise NotImplementedError(f"Not implemented for {pipeline}")

    print(
        f"\t \t  ------- Finished qualitative evaluation {pipeline} pipeline: run {run_name} -------"
    )

def quantitative_evaluation(
    patient_id:str,
    evaluation_dict: Dict,
    pipeline: str,
    metadata_dict: Dict,
    odl_backend: ODLBackend,
    run_name: str,
    inference_function: Callable,
    dataloader:DataLoader,
) -> Dict:
    ### Currently in a disgusting state ###
    print(f"------- Evaluating {pipeline} pipeline: run {run_name} -------")
    ## Default error function
    error_function = PSNR()

    if pipeline == "reconstruction":
        inference_name = metadata_dict["architecture_dict"]["reconstruction"]["name"]
        sinogram_transforms = Normalise()
        reconstruction_device = metadata_dict["architecture_dict"]["reconstruction"]["device_name"]
        for (reconstruction, slice_indices) in tqdm(dataloader):
            reconstruction = reconstruction.to(reconstruction_device)
            sinogram = odl_backend.get_sinogram(reconstruction)
            sinogram = sinogram_transforms(sinogram)
            with torch.no_grad():
                if inference_name == "backprojection":
                    approximated_reconstruction = inference_function(sinogram)

                elif inference_name == "filtered_backprojection":
                    ### Ugly conversions to numpy needed (are they really?) for fbp with odl backend
                    sinogram = sinogram[0, 0].detach().cpu().numpy()
                    approximated_reconstruction = np.asarray(inference_function(sinogram))
                    approximated_reconstruction = (
                        torch.from_numpy(approximated_reconstruction)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .to(reconstruction_device)
                    )

                elif inference_name == "lpd":
                        approximated_reconstruction, _ = inference_function(sinogram)

                elif inference_name == "fourier_filtering":
                    filtered_sinogram: torch.Tensor = inference_function(sinogram)
                    approximated_reconstruction = odl_backend.get_reconstruction(
                        filtered_sinogram.unsqueeze(0)
                    )

                else:
                    raise NotImplementedError(f"Inference not implemented for {inference_name}")

            for batch_index in range(approximated_reconstruction.size()[0]):
                evaluation_dict[patient_id][int(slice_indices[batch_index])] = error_function(
                    approximated_reconstruction[batch_index], reconstruction[batch_index]
                    ).item()

    else:
        raise NotImplementedError

    return evaluation_dict

def evaluate_metadata_file(
    metadata_file_path: pathlib.Path,
    quantitative: bool,
    experiment_models_folder_path: pathlib.Path,
):
    print(f"\t  Evaluating metadata at path: {metadata_file_path}")
    ## Unpacking metadata
    metadata_dict = dict(json.load(open(metadata_file_path)))
    pipeline = metadata_file_path.parent.parent.stem
    experiment_folder_name = metadata_file_path.parent.stem
    run_name = metadata_file_path.stem
    print(
        f"Running {pipeline} pipeline for {experiment_folder_name} experiment folder: experience {run_name} running on {args.platform}"
    )

    ## Unpacking dicts
    odl_backend = ODLBackend()

    if pipeline == "reconstruction":
        ## Instanciate backend
        scan_parameter_dict = metadata_dict["scan_parameter_dict"]
        odl_backend.initialise_odl_backend_from_metadata_dict(scan_parameter_dict)

    data_feeding_dict = metadata_dict["data_feeding_dict"]

    ## Sanity checks
    check_metadata(metadata_dict, metadata_file_path, verbose=False)

    ## Get inference function
    inference_function = get_inference_function(
        metadata_dict, pipeline, odl_backend, experiment_models_folder_path
    )

    ## Transforms
    transforms = {
        "reconstruction_transforms": Compose([ToFloat(), Normalise()]),
        "mask_transforms": Compose([ToFloat()]),
    }


    for patient_name in patients_list:

        ## Dataset and Dataloader
        lidc_idri_dataset = LIDC_IDRI(
            DATASET_PATH,
            str(pipeline),
            odl_backend,
            data_feeding_dict["training_proportion"],
            False,
            False,
            transform=transforms,
            patient_list = [patient_name])

        if quantitative:
            dataloader = DataLoader(
            lidc_idri_dataset,
            1,
            shuffle=False,
            drop_last=False,
            num_workers=data_feeding_dict["num_workers"],
        )

            results_file_path = pathlib.Path(f"results/{pipeline}/{experiment_folder_name}/{run_name}.json")
            results_file_path.parent.mkdir(exist_ok=True, parents=True)
            if results_file_path.is_file():
                evaluation_dict = json.load(open(results_file_path))
            else:
                evaluation_dict = {}

            if patient_name in evaluation_dict:
                print(f"Patient {patient_name} already evaluated, passing...")
            else:
                evaluation_dict[patient_name] = {}

                evaluation_dict = quantitative_evaluation(
                    patient_name,
                    evaluation_dict,
                    pipeline,
                    metadata_dict,
                    odl_backend,
                    run_name,
                    inference_function,
                    dataloader,
                )  # type:ignore

                with open(results_file_path, "w") as out_file:
                    json.dump(evaluation_dict, out_file, indent=4)
        else:
            image_writer = PyPlotImageWriter(
                pathlib.Path(f"images/{pipeline}/{experiment_folder_name}/{run_name}")
            )
            qualitative_evaluation(
                metadata_dict,
                pipeline,
                odl_backend,
                run_name,
                inference_function,
                lidc_idri_dataset,
                "LIDC-IDRI-0222",
                50,
                image_writer,
                transforms,
            )


def evaluate_experiment_folder(
    experiment_folder_path: pathlib.Path,
    quantitative: bool,
    models_folder_path: pathlib.Path,
):
    display_str = f"Evaluating experiment folder {experiment_folder_path.stem}"
    print("-" * len(display_str))
    print(f"{display_str}")
    print("-" * len(display_str))
    for metadata_file_path in experiment_folder_path.glob("*"):
        evaluate_metadata_file(metadata_file_path, quantitative, models_folder_path)


def evaluate_pipeline(
    pipeline_name: str,
    metadata_folder_path: pathlib.Path,
    quantitative: bool,
    models_folder_path: pathlib.Path,
):
    print(f"Evaluating pipeline {pipeline_name}")
    pipeline_folder_name = metadata_folder_path.joinpath(pipeline_name)
    for experiment_folder_path in pipeline_folder_name.glob("*"):
        evaluate_experiment_folder(
            experiment_folder_path, quantitative, models_folder_path
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", required=True)

    parser.add_argument("--pipeline_evaluation", action="store_true")
    parser.add_argument("--pipeline", required=False)

    parser.add_argument("--file_evaluation", dest="pipeline_evaluation", action="store_false"    )
    parser.add_argument("--metadata_path", required=False)

    parser.add_argument("--quantitative", action="store_true")
    parser.add_argument("--qualitative", dest="quantitative", action="store_false")

    parser.add_argument("--patients_list", default=["LIDC-IDRI-0088"])
    parser.set_defaults(quantitative=True)
    args = parser.parse_args()

    print(f"Running code on {args.platform}")
    ## Unpacking paths
    paths_dict = dict(json.load(open("paths_dict.json")))[args.platform]
    MODELS_PATH = pathlib.Path(paths_dict["MODELS_PATH"])
    DATASET_PATH = pathlib.Path(paths_dict["DATASET_PATH"])

    patients_list = args.patients_list

    if args.pipeline_evaluation:
        evaluate_pipeline(
            args.pipeline,
            pathlib.Path("metadata_folder"),
            args.quantitative,
            MODELS_PATH,
        )
    else:
        metadata_file_path = pathlib.Path(args.metadata_path)
        pipeline = metadata_file_path.parent.parent.stem
        experiment_folder_name = metadata_file_path.parent.stem
        run_name = metadata_file_path.stem
        evaluate_metadata_file(
            metadata_file_path,
            args.quantitative,
            MODELS_PATH,
        )
