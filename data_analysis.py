import pathlib
import statistics
from typing import Dict
import pandas as pd
import json

from utils import load_json
from utils import unpack_hparams

METADATA_PATH = pathlib.Path('metadata_folder')
RESULTS_PATH = pathlib.Path('results')

def process_results_file(results_file_path:pathlib.Path, results_dataframe:pd.DataFrame) -> pd.DataFrame:
    assert results_file_path.is_file(), f'There is no result file at {results_file_path}'
    pipeline = results_file_path.parent.parent.stem
    experiment_folder_name = results_file_path.parent.stem
    run_name = results_file_path.stem
    result_dict = json.load(open(results_file_path, "r"))

    metadata_file_path = METADATA_PATH.joinpath(f'{pipeline}/{experiment_folder_name}/{run_name}.json')
    if not metadata_file_path.is_file():
        print(f'There is no metadata file at {metadata_file_path}')
        return results_dataframe
    metadata_dict = dict(json.load(open(metadata_file_path)))
    hparams = unpack_hparams(metadata_dict)

    entry = {
        'pipeline':pipeline,
        'experiment_folder_name':experiment_folder_name,
        'run_name':run_name,
        }
    for key, value in hparams.items():
        entry[key] = value

    for patient_name, patient_dict in result_dict.items():
        entry[f'{patient_name}'] = statistics.mean(patient_dict.values())

    if results_dataframe.empty:
        return results_dataframe.append(entry, ignore_index=True) #type:ignore

    else:
        dataframe = pd.DataFrame.from_dict({key:[value] for key, value in entry.items()})
        return pd.concat([results_dataframe, dataframe])

def create_dataframe(results_dataframe_path:pathlib.Path):
        results_dataframe = pd.DataFrame()
        for result_path in list(RESULTS_PATH.glob('*')):
            for pipeline_path in list(result_path.glob('*')):
                for experiment_path in list(pipeline_path.glob('*')):
                    results_dataframe = process_results_file(experiment_path, results_dataframe)

        print(results_dataframe)
        results_dataframe.to_csv(results_dataframe_path)


def compare_3d_best(dataframe:pd.DataFrame):
    longest_name = int(dataframe['run_name'].astype(bytes).str.len().max())

    for experiment_folder_name in ['6_percent_measurements', '25_percent_measurements', '100_percent_measurements']:
        print(f'Processing folder: {experiment_folder_name}')
        folder_df = dataframe.loc[dataframe['experiment_folder_name'] == experiment_folder_name]
        print('\t' + f' Patient Name  |   Vanilla LPD  | Associated PSNR | First best Name  | Associated PSNR | Second best Name | Associated PSNR | Third best Name  | Associated PSNR')
        for patient_name in ["LIDC-IDRI-0772", "LIDC-IDRI-0893", "LIDC-IDRI-0900", "LIDC-IDRI-1002"]:
            first_best_row  = folder_df[patient_name].nlargest(3).keys()[0]
            second_best_row = folder_df[patient_name].nlargest(3).keys()[1]
            third_best_row = folder_df[patient_name].nlargest(3).keys()[2]

            best_name = folder_df.loc[first_best_row]['run_name']
            scnd_name = folder_df.loc[second_best_row]['run_name']
            thrd_name = folder_df.loc[third_best_row]['run_name']

            best_value  = folder_df.loc[first_best_row][patient_name]
            second_best_value = folder_df.loc[second_best_row][patient_name]
            third_best_value  = folder_df.loc[third_best_row][patient_name]

            vanilla_lpd = '2d_5it_cnn_cnn'
            vanilla_lpd_value = folder_df.loc[dataframe['run_name'] == vanilla_lpd][patient_name].values[0]
            print('\t'+ f'{patient_name} | {vanilla_lpd} |      {vanilla_lpd_value:.2f}      | {best_name}{" "*(longest_name- len(best_name))} |      {best_value:.2f}      | {scnd_name}{" "*(longest_name- len(scnd_name))} |      {second_best_value:.2f}      | {thrd_name}{" "*(longest_name- len(thrd_name))} |      {third_best_value:.2f}'\
                    )


def compare_vanilla_to_1d(dataframe:pd.DataFrame):
    for experiment_folder_name in ['6_percent_measurements', '25_percent_measurements', '100_percent_measurements']:
        print(f'Processing folder: {experiment_folder_name}')
        folder_df = dataframe.loc[dataframe['experiment_folder_name'] == experiment_folder_name]
        run_name = '1d_5it_unet_unet'
        vanilla_lpd_run_name = '2d_5it_cnn_cnn'

        print('\t' + f' Patient Name  | Vanilla LPD PSNR | {run_name} PSNR ')
        for patient_name in ["LIDC-IDRI-0772", "LIDC-IDRI-0893", "LIDC-IDRI-0900", "LIDC-IDRI-1002"]:
            vanilla_lpd_value = folder_df.loc[dataframe['run_name'] == vanilla_lpd_run_name][patient_name].values[0]
            current_run_value = folder_df.loc[dataframe['run_name'] == run_name][patient_name].values[0]
            print('\t'+ f'{patient_name} |       {vanilla_lpd_value:.2f}      |     {current_run_value:.2f}     ' )

if __name__ == '__main__':
    dataframe_path = pathlib.Path('data_analysis.csv')
    if not dataframe_path.is_file():
        create_dataframe(dataframe_path)

    dataframe = pd.read_csv(dataframe_path)

    compare_vanilla_to_1d(dataframe)


