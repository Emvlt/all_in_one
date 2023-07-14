import argparse
import pathlib
from datetime import datetime

import json
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # type:ignore
from kymatio.torch import Scattering2D
import torch

from datasets import LIDC_IDRI
from backends.odl import ODLBackend
from train_functions import train_reconstruction_network, train_segmentation_network
from utils import PyPlotImageWriter
from metadata_checker import check_metadata
from transforms import Normalise, ToFloat  # type:ignore

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
    training_lidc_idri_dataset = LIDC_IDRI(
        DATASET_PATH,
        str(pipeline),
        odl_backend,
        data_feeding_dict["training_proportion"],
        data_feeding_dict["train"],
        data_feeding_dict["is_subset"],
        transform=transforms,
    )

    training_dataloader = DataLoader(
        training_lidc_idri_dataset,
        training_dict["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=training_dict["num_workers"],
    )

    image_writer = PyPlotImageWriter(
        pathlib.Path(f"images") / pipeline / experiment_folder_name / run_name
    )

    rec = next(iter(training_dataloader))

    t0 = datetime.now()
    scattering = Scattering2D(
        J=3, shape=(512, 512), L=8, max_order=3  # scale  # n_angles
    ).to(torch.device("cuda:1"))
    t1 = datetime.now()
    print(f"Elapsed Time for Initialisation : {t1-t0}")

    t0 = datetime.now()
    print(scattering(rec.to(torch.device("cuda:1"))).size())
    t1 = datetime.now()
    print(f"Elapsed Time for Inference : {t1-t0}")
