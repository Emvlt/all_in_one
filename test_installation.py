import torch
import numpy as np
import json
from torch.utils.data import Dataset

from backends.odl import ODLBackend
import json
from torchvision.transforms import Compose
from torch.utils.data import DataLoader

from datasets import LIDC_IDRI
from backends.odl import ODLBackend
from train_functions import train_end_to_end, train_joint, train_reconstruction_network, train_segmentation_network
from utils import check_metadata
from transforms import Normalise, PoissonSinogramTransform

import torch.nn as nn
import torch

from torch.utils.tensorboard import SummaryWriter #type:ignore

from models import LearnedPrimalDual, Unet2D512 #type:ignore

print(f'Is cuda available? : {torch.cuda.is_available()}')
