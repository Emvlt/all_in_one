import argparse
import pathlib
from typing import List, Dict

import json
import matplotlib.pyplot as plt

def load_patient_run(path_to_run:pathlib.Path, patient_id:str) -> List:
    run_result:Dict[str, List] = json.load(open(path_to_run, 'r'))
    return run_result[patient_id]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--platform', required=True)
    parser.add_argument('--pipeline', required=True)
    parser.add_argument('--experiment_folder', required=True)
    parser.add_argument('--run_names', default=['lpd_1_layer', 'backprojection', 'filtered_backprojection_hann_filter'])
    parser.add_argument('--patient_list', default=['LIDC-IDRI-0893', 'LIDC-IDRI-1002'])
    args = parser.parse_args()

    print(f'Running code on {args.platform}')
    ## Unpacking paths
    paths_dict = dict(json.load(open('paths_dict.json')))[args.platform]
    MODELS_PATH = pathlib.Path(paths_dict['MODELS_PATH'])
    RUNS_PATH = pathlib.Path(paths_dict['RUNS_PATH'])
    DATASET_PATH = pathlib.Path(paths_dict['DATASET_PATH'])

    results_path = pathlib.Path(f'results/{args.pipeline}/{args.experiment_folder}')
    path_to_plots_folder = pathlib.Path(f'plots/{args.pipeline}/{args.experiment_folder}')
    path_to_plots_folder.mkdir(exist_ok=True, parents=True)
    for patient_name in args.patient_list:
        for run_name in args.run_names:
            path_to_run = results_path.joinpath(f'{run_name}').with_suffix('.json')
            patient_run = load_patient_run(path_to_run, patient_name)
            plt.plot(patient_run, label=run_name)
        plt.legend()
        plt.savefig(path_to_plots_folder.joinpath(f'{patient_name}.jpg'))
        plt.clf()
