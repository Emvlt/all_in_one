import torch.nn as nn
import torch
from typing import Dict
from geometry_utils import generate_geometry
import matplotlib.pyplot as plt

def weights_init(module:nn.Module):
    if isinstance(module, nn.Conv2d):
        shape = module.weight.shape
        lim = torch.sqrt(6/(shape[0] + shape[1])/shape[2]/shape[3])
        module.weight.data.uniform_(-lim, lim)
        module.bias.data.fill_(0)

class CnnBlock(nn.Module):
    def __init__(self, n_filters:int, n_channels_input:int, n_channels_output:int) -> None:
        super(CnnBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(n_channels_input, n_filters, 3, padding=1),
            nn.PReLU(num_parameters=n_filters, init=0.0),
            nn.Conv2d(n_filters, n_filters, 3, padding=1),
            nn.PReLU(num_parameters=n_filters, init=0.0),
            nn.Conv2d(n_filters, n_channels_output, 3, padding=1))
        weights_init(self.block)

    def forward(self, input_tensor:torch.Tensor) -> torch.Tensor:
        return self.block(input_tensor)

## TODO: get operator types for hints
class Iteration(nn.Module):
    def __init__(self, pytorch_operator, pytorch_operator_adj, operator_norm, n_filters = 32):
        super(Iteration, self).__init__()
        self.op = pytorch_operator
        self.op_adj = pytorch_operator_adj
        self.operator_norm = operator_norm

        self.primal_block = CnnBlock(n_filters, 2, 1)
        self.dual_block   = CnnBlock(n_filters, 3, 1)

    def forward(self, primal:torch.Tensor, dual:torch.Tensor, input_sinogram:torch.Tensor):
        # dual block
        evalop:torch.Tensor = self.op(primal) / self.operator_norm
        inp = torch.cat([dual, evalop, input_sinogram / self.operator_norm ], dim=1)
        dual = dual + self.dual_block(inp)

        # primal block
        evalop = self.op_adj(dual) / self.operator_norm
        inp = torch.cat([primal, evalop], dim=1)
        primal = primal + self.primal_block(inp)

        return primal, dual

class IterativeNetwork(nn.Module):
    def __init__(self, device:torch.device, geometry_parameters:Dict, architecture_parameters:Dict, training_parameters:Dict):
        super().__init__()
        pytorch_operator, pytorch_operator_adj, operator_norm = generate_geometry(device, geometry_parameters)
        self.n_iterations = architecture_parameters["n_iterations"]
        self.primal = torch.zeros((training_parameters["batch_size"], 1) + pytorch_operator.operator.domain.shape).to(device)
        self.dual   = torch.zeros((training_parameters["batch_size"], 1) + pytorch_operator_adj.operator.domain.shape).to(device)
        self.iteration_modules = torch.nn.ModuleDict({f'iteration_{i}':Iteration(pytorch_operator, pytorch_operator_adj, operator_norm) for i in range(self.n_iterations)} ).to(device)

    def forward(self, input_tensor:torch.Tensor):
        primal = self.primal.clone()
        dual = self.dual.clone()

        for i in range(self.n_iterations):
            primal, dual = self.iteration_modules[f'iteration_{i}'](primal, dual, input_tensor)
        return primal[:, 0:1, :]
