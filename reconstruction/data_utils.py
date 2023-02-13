from constants import MU_WATER, PHOTONS_PER_PIXEL, DATASETPATH
import odl
from typing import Dict
from torch.utils.data import Dataset
import torch
import odl.contrib.torch as odl_torch
##TODO :
## Add check for operator is None and phantom
## Check size consistency
def mayo_transform(input_tensor:torch.Tensor):
    transformed_tensor = torch.exp(-MU_WATER * input_tensor)
    noisy_tensor = torch.poisson(transformed_tensor * PHOTONS_PER_PIXEL)
    noisy_tensor = torch.maximum(noisy_tensor, torch.ones(noisy_tensor.size())) / PHOTONS_PER_PIXEL
    log_noisy = - torch.log(noisy_tensor) / MU_WATER
    return log_noisy

##TODO :
# Add checks for dimension, modality and dataset_names arguments
# Check for consistency between parameters
class ProjectDataset(Dataset):
    def __init__(self, device : torch.cuda.device, dataset_parameters:Dict, mode:str, transform=None) -> None:
        self.device = device

        self.path_to_data   = DATASETPATH.joinpath(f"{dataset_parameters['dimension']}/{dataset_parameters['dataset_name']}/{mode}/sinograms")
        self.path_to_target = DATASETPATH.joinpath(f"{dataset_parameters['dimension']}/{dataset_parameters['dataset_name']}/{mode}/phantoms")

        for path in [self.path_to_data, self.path_to_target]:
            if not path.is_dir() :
                raise FileExistsError(f"The path {path} is not a directory.")

        self.dataset_size  = len(list(self.path_to_data.glob('*.pt')))
        self.target_length = len(list(self.path_to_target.glob('*.pt')))
        assert(self.dataset_size == self.target_length)

        self.transform = transform

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        sinogram = torch.load(self.path_to_data.joinpath(f'{index}.pt'))[0]
        phantom  = torch.load(self.path_to_target.joinpath(f'{index}.pt'))[0]

        if self.transform is not None:
            sinogram = self.transform(sinogram)

        return sinogram.to(self.device), phantom.to(self.device)

