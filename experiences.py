import argparse
import pathlib
from datetime import datetime
from typing import Dict

import json
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # type:ignore

from datasets import LIDC_IDRI
from backends.odl import ODLBackend
from train_functions import train_reconstruction_network, train_segmentation_network
from utils import PyPlotImageWriter
from metadata_checker import check_metadata
from transforms import Normalise, ToFloat  # type:ignore

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
        "dose":training_dict["dose"],
        "dual_loss_weighting":training_dict["dual_loss_weighting"],
        "reconstruction_loss":training_dict["reconstruction_loss"],
        "sinogram_loss":training_dict["sinogram_loss"],
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
        hparams["angle_partition_min_pt"]=scan_parameter_dict['angle_partition_dict']['min_pt']
        hparams["angle_partition_max_pt"]=scan_parameter_dict['angle_partition_dict']['max_pt']
        hparams["angle_partition_shape"]=scan_parameter_dict['angle_partition_dict']['shape']
        hparams["detector_partition_min_pt"]=scan_parameter_dict['detector_partition_dict']['min_pt']
        hparams["detector_partition_max_pt"]=scan_parameter_dict['detector_partition_dict']['max_pt']
        hparams["detector_partition_shape"]=scan_parameter_dict['detector_partition_dict']['shape']
        hparams["src_radius"]=scan_parameter_dict['geometry_dict']['src_radius']
        hparams["det_radius"]=scan_parameter_dict['geometry_dict']['det_radius']
        hparams["beam_geometry"]=scan_parameter_dict['geometry_dict']['beam_geometry']

    return hparams

VERBOSE_DICT ={
    'holly-b':True,
    'hpc':False
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_path", required=True)
    parser.add_argument("--platform", required=True)
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

    ## Sanity checks
    check_metadata(metadata_dict)

    ## Instanciate backend
    odl_backend = ODLBackend()
    try:
        scan_parameter_dict = metadata_dict["scan_parameter_dict"]
        odl_backend.initialise_odl_backend_from_metadata_dict(scan_parameter_dict)
    except KeyError:
        print("No scanning dict in metadata, passing...")

    ## Transforms
    transforms = {
        "reconstruction_transforms": Compose([ToFloat(), Normalise()]),
        "mask_transforms": Compose([ToFloat()]),
    }

    ## Dataset and Dataloader
    if data_feeding_dict["is_subset"]:
        training_lidc_idri_dataset = LIDC_IDRI(
            DATASET_PATH,
            str(pipeline),
            odl_backend,
            data_feeding_dict["training_proportion"],
            data_feeding_dict["train"],
            data_feeding_dict["is_subset"],
            transform=transforms,
            subset=data_feeding_dict['subset']
        )
    else:
        training_lidc_idri_dataset = LIDC_IDRI(
            DATASET_PATH,
            str(pipeline),
            odl_backend,
            data_feeding_dict["training_proportion"],
            data_feeding_dict["train"],
            data_feeding_dict["is_subset"],
            transform=transforms
        )

    training_dataloader = DataLoader(
        training_lidc_idri_dataset,
        data_feeding_dict["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=data_feeding_dict["num_workers"],
    )

    image_writer = PyPlotImageWriter(
        pathlib.Path(f"images") / pipeline / experiment_folder_name / run_name
    )

    run_writer = SummaryWriter(
        log_dir=pathlib.Path(RUNS_PATH) / pipeline / experiment_folder_name / run_name
    )
    ### Format hyperparameters for registration
    hparams = unpack_hparams(metadata_dict)
    run_writer.add_hparams(hparams, metric_dict = {})
    models_path = pathlib.Path(MODELS_PATH)

    t0 = datetime.now()

    if pipeline == "reconstruction":
        train_reconstruction_network(
            odl_backend=odl_backend,
            architecture_dict=architecture_dict,
            training_dict=training_dict,
            train_dataloader=training_dataloader,
            image_writer=image_writer,
            run_writer=run_writer,
            save_folder_path=models_path,
            verbose=VERBOSE_DICT[args.platform]
        )
    elif pipeline == "segmentation":
        train_segmentation_network(
            odl_backend=odl_backend,
            architecture_dict=architecture_dict,
            training_dict=training_dict,
            train_dataloader=training_dataloader,
            image_writer=image_writer,
            run_writer=run_writer,
            save_folder_path=models_path,
        )

    else:
        raise ValueError(
            f"Wrong type value, must be fourier_filter, joint, sequential or end_to_end, not {pipeline}"
        )  # type:ignore

    t1 = datetime.now()

    print(f"Elapsed Time : {t1-t0}")
    print("Training Finished \u2713 ")
