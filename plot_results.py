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
    "ramp_vs_lpd": [
        "lpd_1_layer_full_dataset_L1_sinogram_loss", "ramp_filter"
        ],
    "lpd_methods_comparison": [
        "lpd_1_layer_full_dataset_L1_sinogram_loss",
        "lpd_1_layer_full_dataset",
        "lpd_1_layer_2d_sinogram_filtering_full_dataset",
        "filtered_backprojection_hann_filter",
    ],
    "evaluation_comparison":[
        '1d_lpd_sinogram_optimisation_L1',
        '1d_lpd_sinogram_optimisation_MSE',
        '1d_lpd',
        'vanilla_lpd',
        'vanilla_lpd_sinogram_optimisation_L1',
        'vanilla_lpd_sinogram_optimisation_MSE',
    ],
    "1d_architecture_exploration":[
        '1d_1it_cnn_cnn',
        '1d_1it_cnn_unet',
        '1d_1it_unet_cnn',
        '1d_1it_unet_unet',
        '1d_5it_cnn_cnn',
        '1d_5it_cnn_unet',
        '1d_5it_unet_cnn',
        '1d_5it_unet_unet',
    ],
    "2d_architecture_exploration":[
        '2d_1it_cnn_cnn',
        '2d_1it_cnn_unet',
        '2d_1it_unet_cnn',
        '2d_1it_unet_unet',
        '2d_5it_cnn_cnn',
        '2d_5it_cnn_unet',
        '2d_5it_unet_cnn',
        '2d_5it_unet_unet',
    ],

}


def load_patient_run(path_to_run: pathlib.Path, patient_id: str) -> List:
    run_result: Dict[str, Dict] = json.load(open(path_to_run, "r"))
    patient_run_dict = run_result[patient_id]
    n_slices = len(patient_run_dict)
    return [patient_run_dict[str(i)] for i in range(n_slices)]

def assess_setting(patient_name: str,pipeline: str,experiment_folder:str,experiment_tracker: List):
    experiment_save_path = PATH_TO_PLOTS.joinpath(f"{experiment_folder}")
    experiment_save_path.mkdir(exist_ok=True, parents=True)

    fig = plt.figure(1)
    ax = fig.add_subplot(111)

    for experiment_name in experiment_tracker:
        path_to_run = RESULTS_PATH.joinpath(
            f"{pipeline}/{experiment_folder}/{experiment_name}.json"
        )
        patient_run = load_patient_run(path_to_run, patient_name)
        ### Format run

        ### Plot run
        ax.plot(patient_run, label=f'{experiment_name}_{statistics.mean(patient_run)}')
        print("\t" + f"Run name: {experiment_folder}/{experiment_name}")
        print("\t \t" + f"Average PSNR: {statistics.mean(patient_run)}")

    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,-0.1))
    ax.set_title(f'{args.experiment_name}')
    plt.xlabel("Slice index")
    plt.ylabel("PSNR")
    plt.savefig(experiment_save_path.joinpath(f"{patient_name}.jpg"), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()

def assess_pipeline(patient_name: str, pipeline: str, experiment_tracker: List):
    for experiment_folder in PIPELINE_FOLDERS[pipeline]:
        assess_setting(patient_name, pipeline, experiment_folder, experiment_tracker)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", required=False, default='holly-b')

    parser.add_argument("--experiment_name", required=True)

    parser.add_argument("--pipeline", required=False, default ='reconstruction')

    parser.add_argument("--pipeline_evaluation", action="store_true")

    parser.add_argument("--setting_evaluation", dest="pipeline_evaluation", action="store_false")
    parser.add_argument("--setting", required=False)

    parser.add_argument("--patient_list", default=["LIDC-IDRI-0772", "LIDC-IDRI-0893", "LIDC-IDRI-0900", "LIDC-IDRI-1002"])
    parser.set_defaults(quantitative=True)


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
        if args.pipeline_evaluation:
            assess_pipeline(patient_name, args.pipeline, experiment_tracker)
        else:
            assess_setting(patient_name, args.pipeline, args.setting, experiment_tracker)
