import argparse
import pathlib
from typing import List, Dict
import statistics

import json
import matplotlib.pyplot as plt

PIPELINE_FOLDERS = {
    "reconstruction": [
        "6_percent_measurements",
        "25_percent_measurements",
        "100_percent_measurements",
    ],
    "segmentation": [
        "from_input_images",
        "6_percent_measurements",
        "25_percent_measurements",
        "100_percent_measurements",
    ],
}

EXPERIMENT_TRACKER = {
    "sinogram_enhancement": [
        "lpd_1_layer_full_dataset",
        "lpd_1_layer_full_dataset_MSE_sinogram_loss",
        "lpd_1_layer_full_dataset_L1_sinogram_loss",
    ],
    "ramp_vs_lpd": ["lpd_1_layer_full_dataset_L1_sinogram_loss", "ramp_filter"],
    "lpd_methods_comparison": [
        "lpd_1_layer_full_dataset_L1_sinogram_loss",
        "lpd_1_layer_full_dataset",
        "lpd_1_layer_2d_sinogram_filtering_full_dataset",
        "filtered_backprojection_hann_filter",
    ],
}


def load_patient_run(path_to_run: pathlib.Path, patient_id: str) -> List:
    run_result: Dict[str, List] = json.load(open(path_to_run, "r"))
    return run_result[patient_id]


def compare_methods(
    patient_name: str,
    results_path: pathlib.Path,
    pipeline: str,
    experiment_tracker: List,
):
    for experiment_folder in PIPELINE_FOLDERS[pipeline]:
        experiment_save_path = PATH_TO_PLOTS.joinpath(f"{experiment_folder}")
        experiment_save_path.mkdir(exist_ok=True, parents=True)
        for experiment_name in experiment_tracker:
            path_to_run = results_path.joinpath(
                f"{pipeline}/{experiment_folder}/{experiment_name}.json"
            )
            patient_run = load_patient_run(path_to_run, patient_name)
            plt.plot(patient_run, label=experiment_name)
            print("\t" + f"Run name: {experiment_folder}/{experiment_name}")
            print("\t \t" + f"Average PSNR: {statistics.mean(patient_run)}")
        plt.title(args.experiment_name)
        plt.xlabel("Slice index")
        plt.ylabel("PSNR")
        plt.legend()
        plt.savefig(experiment_save_path.joinpath(f"{patient_name}.jpg"))
        plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", required=True)
    parser.add_argument("--pipeline", required=True)
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--patient_list", default=["LIDC-IDRI-1002"])
    args = parser.parse_args()

    print(f"Running code on {args.platform}")
    ## Unpacking paths
    paths_dict = dict(json.load(open("paths_dict.json")))[args.platform]
    MODELS_PATH = pathlib.Path(paths_dict["MODELS_PATH"])
    DATASET_PATH = pathlib.Path(paths_dict["DATASET_PATH"])

    experiment_tracker = EXPERIMENT_TRACKER[args.experiment_name]
    PATH_TO_PLOTS = pathlib.Path(f"plots/{args.experiment_name}")

    RESULTS_PATH = pathlib.Path("results")

    for patient_name in args.patient_list:
        compare_methods(patient_name, RESULTS_PATH, args.pipeline, experiment_tracker)
