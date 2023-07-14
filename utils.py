from typing import Dict, List
import pathlib

import json
import torch
import matplotlib.pyplot as plt


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

    def write_image_tensor(self, x: torch.Tensor, image_name: str):
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

    def write_line_tensor(self, x: torch.Tensor, image_name: str):
        plt.plot(x.detach().cpu())
        plt.savefig(self.path_to_images_folder.joinpath(image_name))
        plt.clf()
        plt.close()

    def write_kernel_weights(
        self, x: List[torch.Tensor], names: List[str], fig_name: str
    ):
        f, (axs) = plt.subplots(len(names), sharey=True)
        for i, weight in enumerate(x):
            axs[i].plot(weight.detach().cpu())
            axs[i].set_title(names[i])
        plt.savefig(self.path_to_images_folder.joinpath(fig_name))
        plt.clf()
        plt.close()
