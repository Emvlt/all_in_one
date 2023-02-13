from typing import Dict

import torch
from torch.utils.data import  DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from model_definitions import IterativeNetwork
from data_utils import ProjectDataset, mayo_transform
from statistics_utils import collect_statistics, save_experiment_metadata, load_experiment_metadata
from constants import MODELSPATH

def main(time_stamp:str, metadata_dict:Dict):
    geometry_parameters = metadata_dict['geometry_parameters']
    architecture_parameters = metadata_dict['architecture_parameters']
    training_parameters = metadata_dict['training_parameters']
    dataset_parameters = metadata_dict['dataset_parameters']
    device = metadata_dict['device']

    iterative_network = IterativeNetwork(device, geometry_parameters, architecture_parameters, training_parameters)

    test_dataset    = ProjectDataset(device, dataset_parameters, mode='test', transform=mayo_transform)
    test_dataloader = DataLoader(test_dataset, training_parameters['batch_size'], shuffle=False)

    train_dataset    = ProjectDataset(device, dataset_parameters, mode='train', transform=mayo_transform)
    train_dataloader = DataLoader(train_dataset, training_parameters['batch_size'], shuffle=True)

    optimizer = torch.optim.Adam(iterative_network.parameters(), lr=training_parameters['learning_rate'], betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, training_parameters['epochs']*train_dataset.dataset_size)

    L2_loss = torch.nn.MSELoss()

    iterative_network.train()

    for epoch in tqdm(range(training_parameters['epochs'])):
        for i, data in enumerate(train_dataloader):
            sample, target = data[0], data[1]
            optimizer.zero_grad()
            output = iterative_network(sample.to(device))
            loss = L2_loss(output, target.to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()
            collect_statistics(loss.detach().cpu())

        if epoch % 10 == 0:
            plt.matshow(output[0,0].detach().cpu())
            plt.show()


        #torch.save(iterative_network.state_dict(), MODELSPATH.joinpath(f'{time_stamp}.pt'))

if __name__ =='__main__':
    metadata_dict = load_experiment_metadata('50_19_04_Dec_2022')
    main('50_19_04_Dec_2022', metadata_dict)
