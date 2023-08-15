import argparse
from pathlib import Path
from itertools import groupby
from operator import itemgetter
from typing import Dict
from math import nan


from torch.utils.data import DataLoader
import json
import torch
from torchvision.transforms import Compose
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

from metrics import PSNR
from backends.odl import ODLBackend
from datasets import LIDC_IDRI, LIDC_IDRI_SEGMENTATIONS, PatientDataset
from transforms import Normalise, ToFloat  # type:ignore
from train_functions import unpack_architecture_dicts

def load_evaluation_dict(results_file_path:Path) -> Dict:
    if results_file_path.is_file():
        evaluation_df:pd.DataFrame = pd.read_csv(results_file_path)
        evaluation_dict = evaluation_df.to_dict()
    else:
        evaluation_dict = {
                'patient_index':[],
                'slice':[],
                'n_annotations':[],
                'nodule_size_mean':[],
                'nodule_size_stddev':[],
                'segmentation_BCE':[]
            }
        if pipeline == 'joint':
            evaluation_dict['reconstruction_MSE'] = []
            evaluation_dict['reconstruction_PSNR'] = []
    return evaluation_dict

def display_segmentation(
    path_to_img_folder:Path, patient_dataset:PatientDataset, slice_index:torch.Tensor,
    reconstruction_tensor:torch.Tensor, mask_tensor:torch.Tensor, approximated_segmentation:torch.Tensor
    ):
    if (patient_dataset.annotations_dataframe['slice'] == slice_index.item()).any():
        path_to_image = path_to_img_folder.joinpath(f'{slice_index}_segmentation.jpg')
        print(f'Printing image at {path_to_image}')
        targets = torch.cat(
            (
                display_transform(reconstruction_tensor[index, 0].detach().cpu()),
                display_transform(mask_tensor[index, 0].detach().cpu()),
            ),  dim=1)

        approxs = torch.cat(
                (
                    display_transform(reconstruction_tensor[index, 0].detach().cpu()),
                    display_transform(approximated_segmentation[index, 0].detach().cpu()),
                ),  dim=1)

        plt.matshow(torch.cat((targets, approxs), dim =0))
        plt.axis("off")
        plt.savefig(path_to_image, bbox_inches='tight')
        plt.clf()
        plt.close()

parser = argparse.ArgumentParser()
parser.add_argument("--platform", required=False, default='holly-b')

parser.add_argument('--metadata_path', required=False, default='metadata_folder/segmentation/from_input_images/progressive_training_unet.json')

args = parser.parse_args()

### Unpack metadata
paths_dict = dict(json.load(open("paths_dict.json")))[args.platform]
MODELS_PATH = Path(paths_dict["MODELS_PATH"])
DATASET_PATH = Path(paths_dict["DATASET_PATH"])

metadata_file_path = Path(args.metadata_path)
pipeline = metadata_file_path.parent.parent.stem
experiment_folder_name = metadata_file_path.parent.stem
run_name = metadata_file_path.stem

metadata = json.load(open(metadata_file_path))

## Instanciate backend
odl_backend = ODLBackend()
try:
    scan_parameter_dict = metadata["scan_parameter_dict"]
    odl_backend.initialise_odl_backend_from_metadata_dict(scan_parameter_dict)
except KeyError:
    print("No scanning dict in metadata, passing...")

networks = unpack_architecture_dicts(
    metadata['architecture_dict'],
    odl_backend
    )

checkpoint = torch.load(MODELS_PATH.joinpath(f'{pipeline}/{experiment_folder_name}/{run_name}.pth'))

if pipeline == 'segmentation':
    segmentation_network = networks['segmentation']
    segmentation_network.load_state_dict(checkpoint)
    segmentation_network.eval()

elif pipeline == 'joint':
    segmentation_network = networks['segmentation']
    segmentation_network.load_state_dict(checkpoint['segmentation_net'])
    segmentation_network.eval()

    reconstruction_network = networks['reconstruction']
    reconstruction_network.load_state_dict(checkpoint['reconstruction_net'])
    reconstruction_network.eval()

else:
    raise ValueError

device = torch.device(metadata['architecture_dict']['segmentation']['device_name'])

## Define dataset and dataloader
data_feeding_dict = metadata["data_feeding_dict"]
transforms = {
        "reconstruction_transforms": Compose([ToFloat(), Normalise()]),
        "mask_transforms": Compose([ToFloat()]),
    }
sinogram_transforms = Normalise()

## Dataset
nodule_size = 256

query_string = f'{nodule_size} < nodule_size'

dataset = LIDC_IDRI_SEGMENTATIONS(
    path_to_processed_dataset=DATASET_PATH,
    training_proportion = 0.92,
    training = False,
    query_string = query_string,
    transform = transforms
)

print(f'There are {dataset.__len__()} slices in the dataset')

display_transform = Normalise()

segmentation_loss = torch.nn.BCELoss()
psnr_loss = PSNR()

results_file_path = Path(f"results/{pipeline}/{experiment_folder_name}/{run_name}.csv")
results_file_path.parent.mkdir(exist_ok=True, parents=True)

evaluation_dict = load_evaluation_dict(results_file_path)

for patient_index in dataset.patients_list:
    path_to_img_folder = Path(f'images/{pipeline}/{experiment_folder_name}/{patient_index}')
    path_to_img_folder.mkdir(exist_ok=True, parents= True)

    print(f'Processing patient {patient_index}')
    patient_dataset = dataset.get_patient_dataset(patient_index)

    patient_dataloader = DataLoader(
        patient_dataset,
        data_feeding_dict["batch_size"],
        shuffle=data_feeding_dict['shuffle'],
        drop_last=True,
        num_workers=data_feeding_dict["num_workers"],)


    for slice_indices, reconstruction_tensor, mask_tensor in tqdm(patient_dataloader):
        reconstruction_tensor = reconstruction_tensor.to(device)
        mask_tensor = mask_tensor.to(device)
        if pipeline == 'joint':
            with torch.no_grad():
                ## Re-sample
                sinogram = odl_backend.get_sinogram(reconstruction_tensor)
                sinogram = sinogram_transforms(sinogram)  # type:ignore
                ## Reconstruct
                approximated_reconstruction, approximated_sinogram = reconstruction_network(sinogram) #type:ignore
                loss_recontruction = psnr_loss(approximated_reconstruction, reconstruction_tensor)

                approximated_segmentation = segmentation_network(approximated_reconstruction)
                loss_segmentation = segmentation_loss(approximated_segmentation, mask_tensor)

        elif pipeline == 'segmentation':
            with torch.no_grad():
                approximated_segmentation = segmentation_network(reconstruction_tensor)
                loss_segmentation = segmentation_loss(approximated_segmentation, mask_tensor)

            for index, slice_index in enumerate(slice_indices):
                '''
                display_segmentation(
                    path_to_img_folder, patient_dataset, slice_index,
                    reconstruction_tensor, mask_tensor, approximated_segmentation
                    )
                '''
                evaluation_dict['patient_index'].append(patient_index)
                evaluation_dict['slice'].append(slice_index.item())
                evaluation_dict['segmentation_BCE'].append(segmentation_loss(approximated_segmentation[index], mask_tensor[index]).item())

                slice_annotations = patient_dataset.annotations_dataframe['slice'] == slice_index.item()
                if slice_annotations.any():
                    rows = patient_dataset.annotations_dataframe[slice_annotations]
                    evaluation_dict['nodule_size_mean'].append(rows['nodule_size'].mean())
                    evaluation_dict['nodule_size_stddev'].append(rows['nodule_size'].std())
                    evaluation_dict['n_annotations'].append(len(rows))
                else:
                    evaluation_dict['nodule_size_mean'].append(nan)
                    evaluation_dict['nodule_size_stddev'].append(nan)
                    evaluation_dict['n_annotations'].append(nan)
        else:
            raise ValueError

    print(pd.DataFrame.from_dict(evaluation_dict))

pd.DataFrame.from_dict(evaluation_dict).to_csv(results_file_path)
