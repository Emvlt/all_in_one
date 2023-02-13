from typing import Dict

import torch
from torch.utils.data import  DataLoader
from tqdm import tqdm

from model_definitions import IterativeNetwork
from data_utils import ProjectDataset, mayo_transform
from statistics_utils import collect_statistics
from constants import MODELSPATH

def LPD(metadata_dict:Dict):
    geometry_parameters = metadata_dict['geometry_parameters']
    architecture_parameters = metadata_dict['architecture_parameters']
    training_parameters = metadata_dict['training_parameters']
    dataset_parameters = metadata_dict['dataset_parameters']
    iterative_network = IterativeNetwork(geometry_parameters, architecture_parameters, training_parameters)
    device = torch.device(metadata_dict['device'])
    iterative_network.to(device)

    test_dataset    = ProjectDataset(dataset_parameters, mode='test', transform=mayo_transform)
    test_dataloader = DataLoader(test_dataset, training_parameters['batch_size'], shuffle=False)

    train_dataset    = ProjectDataset(dataset_parameters, mode='train', transform=mayo_transform)
    train_dataloader = DataLoader(train_dataset, training_parameters['batch_size'], shuffle=True)

    optimizer = torch.optim.Adam(iterative_network.parameters(), lr=training_parameters['learning_rate'], betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, training_parameters['epochs']*train_dataset.dataset_size)

    L2_loss = torch.nn.MSELoss()

    iterative_network.train()

    for epoch in tqdm(range(training_parameters['epochs'])):
        for i, data in enumerate(train_dataloader):
            sample, target = data[0], data[1]
            optimizer.zero_grad()
            output = iterative_network(sample)
            loss = L2_loss(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(iterative_network.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if i % 100 == 0:
                collect_statistics(loss.detach().cpu())

        torch.save(iterative_network.state_dict(), MODELSPATH.joinpath(training_parameters["time_stamp"]))