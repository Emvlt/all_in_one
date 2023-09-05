import argparse
import pathlib

import json
from torchvision.transforms import Compose
import matplotlib.pyplot as plt

from transforms import Normalise, ToFloat  # type:ignore
from datasets import LIDC_IDRI

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_path", required=False, default = 'metadata_folder/joint/6_percent_measurements/lpd_unet_0.json')
    parser.add_argument("--platform", required=False, default='holly-b')
    args = parser.parse_args()

    ## Unpacking paths
    paths_dict = dict(json.load(open("paths_dict.json")))[args.platform]
    MODELS_PATH = pathlib.Path(paths_dict["MODELS_PATH"])
    RUNS_PATH = pathlib.Path(paths_dict["RUNS_PATH"])
    DATASET_PATH = pathlib.Path(paths_dict["DATASET_PATH"])

    ## Unpacking metadata
    metadata_path = pathlib.Path(args.metadata_path)
    pipeline = metadata_path.parent.parent.stem
    experiment_folder_name = metadata_path.parent.stem
    run_name = metadata_path.stem
    print(
        f"Running {pipeline} pipeline for {experiment_folder_name} experiment folder: experience {run_name} running on {args.platform}"
    )

    metadata_dict = dict(json.load(open(metadata_path)))
    data_feeding_dict = metadata_dict["data_feeding_dict"]
    training_dict = metadata_dict["training_dict"]
    architecture_dict = metadata_dict["architecture_dict"]

    transforms = {
        "reconstruction_transforms": Compose([ToFloat(), Normalise()]),
        "mask_transforms": Compose([ToFloat()])
    }
    if pipeline == 'reconstruction':
        query_string = None
    else:
        query_string = f'{256} < nodule_size'
    dataset = LIDC_IDRI(
            path_to_processed_dataset=DATASET_PATH,
            training_proportion = data_feeding_dict['training_proportion'],
            training = True,
            pipeline=pipeline,
            query_string=query_string,
            transform = transforms
        )
    print(dataset)
    print(dataset.get_all_patient_slices('LIDC-IDRI-0001'))
    '''patient_dataset = dataset.get_patient_dataset('LIDC-IDRI-0020')
    print(patient_dataset)
    index = 100
    rec = patient_dataset.compute_reconstruction_tensor(index)
    mask = patient_dataset.compute_mask_tensor(index)
    print(mask.size())
    plt.matshow(mask[0].detach().cpu())
    plt.savefig(f'm_{index}.jpg')
    plt.clf()'''


