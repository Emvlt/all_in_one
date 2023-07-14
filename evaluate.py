import argparse
import pathlib
from typing import Dict, Callable, List

import torch
import json
from torchvision.transforms import Compose
import numpy as np

from metrics import PSNR  # type:ignore
from models import LearnedPrimalDual, load_network, FourierFilteringModule
from datasets import LIDC_IDRI
from backends.odl import ODLBackend
from utils import PyPlotImageWriter
from metadata_checker import check_metadata
from transforms import Normalise, ToFloat  # type:ignore


def get_inference_function(
    metadata_dict: Dict,
    pipeline: str,
    odl_backend: ODLBackend,
    experiment_models_folder_path: pathlib.Path,
) -> Callable:

    architecture_dict = metadata_dict["architecture_dict"]
    if pipeline == "reconstruction":
        architecture_name = architecture_dict["reconstruction"]["name"]
        print(f"\t \t  Loading inference function for {architecture_name}")
        if architecture_name == "lpd":
            reconstruction_device = torch.device(
                architecture_dict["reconstruction"]["device_name"]
            )
            lpd_network = LearnedPrimalDual(
                dimension=architecture_dict["reconstruction"]["dimension"],
                odl_backend=odl_backend,
                n_primal=architecture_dict["reconstruction"]["n_primal"],
                n_dual=architecture_dict["reconstruction"]["n_dual"],
                n_iterations=architecture_dict["reconstruction"]["lpd_n_iterations"],
                n_filters_primal=architecture_dict["reconstruction"][
                    "lpd_n_filters_primal"
                ],
                n_filters_dual=architecture_dict["reconstruction"][
                    "lpd_n_filters_dual"
                ],
                fourier_filtering=architecture_dict["reconstruction"][
                    "fourier_filtering"
                ],
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
            reconstruction_device = torch.device(
                architecture_dict["reconstruction"]["device_name"]
            )
            fourier_filtering_module = FourierFilteringModule(
                dimension=architecture_dict["reconstruction"]["dimension"],
                n_measurements=odl_backend.angle_partition_dict["shape"],
                detector_size=odl_backend.detector_partition_dict["shape"],
                device=reconstruction_device,
                filter_name=architecture_dict["reconstruction"]["filter_name"],
                training_mode=False,
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
    dimension: int,
    inference_function: Callable,
    patient_slice_path: pathlib.Path,
    dataset: LIDC_IDRI,
    reconstruction_device: torch.device,
    sinogram_transforms,
    inference_name,
    reconstruction_transforms: Dict,
):
    reconstruction = (
        dataset.get_reconstruction_tensor(patient_slice_path)
        .unsqueeze(0)
        .to(reconstruction_device)
    )
    reconstruction = reconstruction_transforms["reconstruction_transforms"](
        reconstruction
    )
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
        if dimension == 1:
            sinogram = torch.squeeze(sinogram, dim=1)
        with torch.no_grad():
            approximated_reconstruction, _ = inference_function(sinogram)

    elif inference_name == "fourier_filtering":
        if dimension == 1:
            sinogram = torch.squeeze(sinogram, dim=1)
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
            metadata_dict["architecture_dict"]["reconstruction"]["dimension"],
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
    evaluation_dict: Dict,
    pipeline: str,
    metadata_dict: Dict,
    odl_backend: ODLBackend,
    run_name: str,
    inference_function: Callable,
    dataset: LIDC_IDRI,
    transforms,
) -> Dict:
    ### Currently in a disgusting state ###
    print(f"------- Evaluating {pipeline} pipeline: run {run_name} -------")
    ## Default error function
    error_function = PSNR()

    for patient_id in dataset.testing_patients_list:
        print(f"Evaluating patient {patient_id}")
        if patient_id in evaluation_dict:
            print(f"Patient {patient_id} already evaluated, passing...")
        else:
            evaluation_dict[patient_id] = []

            if pipeline == "reconstruction":
                inference_name = metadata_dict["architecture_dict"]["reconstruction"][
                    "name"
                ]
                sinogram_transforms = Normalise()
                reconstruction_device = metadata_dict["architecture_dict"][
                    "reconstruction"
                ]["device_name"]
                patient_indices_list = dataset.get_patient_slices_list(patient_id)
                for patient_slice_path in patient_indices_list:
                    approximated_reconstruction, reconstruction = infer_slice(
                        odl_backend,
                        metadata_dict["architecture_dict"]["reconstruction"][
                            "dimension"
                        ],
                        inference_function,
                        patient_slice_path,
                        dataset,
                        reconstruction_device,
                        sinogram_transforms,
                        inference_name,
                        transforms,
                    )
                    evaluation_dict[patient_id].append(
                        error_function(
                            approximated_reconstruction, reconstruction
                        ).item()
                    )

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
    check_metadata(metadata_dict, verbose=False)

    ## Transforms
    transforms = {
        "reconstruction_transforms": Compose([ToFloat(), Normalise()]),
        "mask_transforms": Compose([ToFloat()]),
    }

    ## Dataset and Dataloader
    lidc_idri_dataset = LIDC_IDRI(
        DATASET_PATH,
        pipeline,
        odl_backend,
        data_feeding_dict["training_proportion"],
        False,
        data_feeding_dict["is_subset"],
        transform=transforms,
        verbose=False,
        patient_list=args.patient_list,
    )

    inference_function = get_inference_function(
        metadata_dict, pipeline, odl_backend, experiment_models_folder_path
    )

    if quantitative:
        results_file_path = pathlib.Path(
            f"results/{pipeline}/{experiment_folder_name}/{run_name}.json"
        )
        if results_file_path.is_file():
            evaluation_dict = json.load(open(results_file_path))
        else:
            evaluation_dict = {}

        evaluation_dict = quantitative_evaluation(
            evaluation_dict,
            pipeline,
            metadata_dict,
            odl_backend,
            run_name,
            inference_function,
            lidc_idri_dataset,
            transforms,
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

    parser.add_argument("--pipeline_evaluation", dest="store_true")
    parser.add_argument("--pipeline", required=False)

    parser.add_argument(
        "--file_evaluation", dest="pipeline_evaluation", action="store_false"
    )
    parser.add_argument("--metadata_path", required=False)

    parser.add_argument("--quantitative", action="store_true")
    parser.add_argument("--qualitative", dest="quantitative", action="store_false")

    parser.add_argument("--patient_list", default=["LIDC-IDRI-0893", "LIDC-IDRI-1002"])
    parser.set_defaults(quantitative=True)
    args = parser.parse_args()

    print(f"Running code on {args.platform}")
    ## Unpacking paths
    paths_dict = dict(json.load(open("paths_dict.json")))[args.platform]
    MODELS_PATH = pathlib.Path(paths_dict["MODELS_PATH"])
    DATASET_PATH = pathlib.Path(paths_dict["DATASET_PATH"])

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
            MODELS_PATH / pipeline / experiment_folder_name,
        )
