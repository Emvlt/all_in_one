from typing import Dict, List
import pathlib

import json
import matplotlib.pyplot as plt

def unpack_hparams(metadata_dict:Dict) -> Dict:
    data_feeding_dict = metadata_dict['data_feeding_dict']
    training_dict = metadata_dict['training_dict']
    architecture_dict = metadata_dict['architecture_dict']
    hparams = {
        ### Data feeding dict unpacking
        "dataset_name":data_feeding_dict["dataset_name"],
        "training_proportion":data_feeding_dict["training_proportion"],
        "is_subset":data_feeding_dict["is_subset"],
        "batch_size":data_feeding_dict["batch_size"],
        "num_workers":data_feeding_dict["num_workers"],
        ### Training dict unpacking
        "learning_rate":training_dict["learning_rate"],
        "n_epochs":training_dict["n_epochs"],
    }
    if training_dict["dual_loss_weighting"] !=0:
        hparams["sinogram_loss"] = training_dict["sinogram_loss"]
    for architecture_name, network_dict in architecture_dict.items():
        hparams[f'{architecture_name}_network'] = network_dict["name"]
        if network_dict["name"] == 'lpd':
            for key, value in network_dict['primal_dict'].items():
                hparams[f"primal_{key}"] = value
            for key, value in network_dict['dual_dict'].items():
                hparams[f"dual_{key}"] = value
            hparams['lpd_n_iterations'] = network_dict['n_iterations']
            for key, value in network_dict['fourier_filtering_dict'].items():
                hparams[f"fourier_filtering_{key}"] = value
        else:
            raise NotImplementedError


    if 'scan_parameter_dict' in metadata_dict.keys():
        ### Scan dict unpacking
        hparams["angle_partition_min_pt"]=metadata_dict['scan_parameter_dict']['angle_partition_dict']['min_pt']
        hparams["angle_partition_max_pt"]=metadata_dict['scan_parameter_dict']['angle_partition_dict']['max_pt']
        hparams["angle_partition_shape"]=metadata_dict['scan_parameter_dict']['angle_partition_dict']['shape']
        hparams["detector_partition_min_pt"]=metadata_dict['scan_parameter_dict']['detector_partition_dict']['min_pt']
        hparams["detector_partition_max_pt"]=metadata_dict['scan_parameter_dict']['detector_partition_dict']['max_pt']
        hparams["detector_partition_shape"]=metadata_dict['scan_parameter_dict']['detector_partition_dict']['shape']
        hparams["src_radius"]=metadata_dict['scan_parameter_dict']['geometry_dict']['src_radius']
        hparams["det_radius"]=metadata_dict['scan_parameter_dict']['geometry_dict']['det_radius']
        hparams["beam_geometry"]=metadata_dict['scan_parameter_dict']['geometry_dict']['beam_geometry']

    return hparams

def load_json(file_path: pathlib.Path):
    if not file_path.is_file():
        raise FileNotFoundError(f"No file found at {file_path}")
    with open(file_path, "r") as file_read:
        file = json.load(file_read)
    return file


def save_json(file_path: pathlib.Path, file: Dict):
    with open(file_path, "w") as file_write:
        json.dump(file, file_write)


class PyPlotImageWriter:
    def __init__(self, path_to_images_folder: pathlib.Path) -> None:
        self.path_to_images_folder = path_to_images_folder
        self.path_to_images_folder.mkdir(parents=True, exist_ok=True)

    def write_image_tensor(self, x, image_name: str):
        x = x.detach().cpu()
        ## unholy
        while len(x.size()) != 2:
            x = x[0]
        plt.matshow(x)
        plt.axis("off")
        plt.savefig(
            self.path_to_images_folder.joinpath(image_name), bbox_inches="tight"
        )
        plt.clf()
        plt.close()

    def write_line_tensor(self, x, image_name: str):
        plt.plot(x.detach().cpu())
        plt.savefig(self.path_to_images_folder.joinpath(image_name))
        plt.clf()
        plt.close()

    def write_kernel_weights(
        self, x: List, names: List[str], fig_name: str
    ):
        f, (axs) = plt.subplots(len(names), sharey=True)
        for i, weight in enumerate(x):
            axs[i].plot(weight.detach().cpu())
            axs[i].set_title(names[i])
        plt.savefig(self.path_to_images_folder.joinpath(fig_name))
        plt.clf()
        plt.close()
