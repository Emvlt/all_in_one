import argparse
from typing import Dict
from datetime import datetime
import json
from constants import METADATAPATH
import model_definitions
from data_utils import ProjectDataset, mayo_transform
from torch.utils.data import  DataLoader

#TODO: create custom class for experience metadata with checks and whatnot + read/dump semantic
def save_experiment_metadata(metadata_dict:Dict):
    time_stamp = datetime.now().strftime("%M_%H_%d_%b_%Y")
    with open(str(METADATAPATH.joinpath(f'{time_stamp}.json')), 'w') as out_file:
        json.dump(metadata_dict, out_file, indent=4)

def load_experiment_metadata(time_stamp:str):
    metadata_dict:Dict = json.load(open(METADATAPATH.joinpath(f'{time_stamp}.json')))
    return metadata_dict

def write_graph(time_stamp:str):
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
    metadata_dict = load_experiment_metadata(time_stamp)
    model = model_definitions.IterativeNetwork(metadata_dict['geometry_parameters'], metadata_dict['architecture_parameters'], metadata_dict['training_parameters'])
    test_dataset    = ProjectDataset(metadata_dict['dataset_parameters'], mode='test', transform=mayo_transform)
    test_dataloader = DataLoader(test_dataset, metadata_dict['training_parameters']['batch_size'], shuffle=False)
    writer.add_graph(model, next(iter(test_dataloader)))

def collect_statistics(loss):
    print(loss)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--time_stamp', required=False, type=str, default='50_19_04_Dec_2022')
    args = parser.parse_args()
    write_graph()