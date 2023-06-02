from typing import Dict

import torch.nn as nn
import torch

from backends.odl import ODLBackend

def weights_init(module:nn.Module):
    if isinstance(module, nn.Conv1d):
        shape = module.weight.shape
        lim = (6/(shape[0] + shape[1])/shape[2])**0.5
        module.weight.data.uniform_(-lim, lim)
        module.bias.data.fill_(0) # type:ignore

    elif isinstance(module, nn.Conv2d):
        shape = module.weight.shape
        lim = (6/(shape[0] + shape[1])/shape[2]/shape[3])**0.5
        module.weight.data.uniform_(-lim, lim)
        module.bias.data.fill_(0) # type:ignore

class CnnBlock(nn.Module):
    def __init__(self, dimension:int, n_filters:int, n_channels_input:int, n_channels_output:int) -> None:
        super(CnnBlock, self).__init__()
        if dimension == 1:
            self.block = nn.Sequential(
                nn.Conv1d(n_channels_input, n_filters, 3, padding=1),
                nn.PReLU(num_parameters=n_filters, init=0.0),
                nn.Conv1d(n_filters, n_filters, 3, padding=1),
                nn.PReLU(num_parameters=n_filters, init=0.0),
                nn.Conv1d(n_filters, n_channels_output, 3, padding=1))
        elif dimension == 2:
            self.block = nn.Sequential(
                nn.Conv2d(n_channels_input, n_filters, 3, padding=1),
                nn.PReLU(num_parameters=n_filters, init=0.0),
                nn.Conv2d(n_filters, n_filters, 3, padding=1),
                nn.PReLU(num_parameters=n_filters, init=0.0),
                nn.Conv2d(n_filters, n_channels_output, 3, padding=1))
        weights_init(self.block)

    def forward(self, input_tensor:torch.Tensor) -> torch.Tensor:
        return self.block(input_tensor)


class Iteration(nn.Module):
    def __init__(self, dimension:int, pytorch_operator, pytorch_operator_adj, operator_norm, n_filters = 32):
        super(Iteration, self).__init__()
        self.dimension = dimension
        self.op = pytorch_operator
        self.op_adj = pytorch_operator_adj
        self.operator_norm = operator_norm
        self.n_filters = n_filters

        ### We implicitely expect the input primal to have one channel
        self.primal_block = CnnBlock(2, self.n_filters, 2, 1)
        if dimension == 1:
            self.dual_block = CnnBlock(dimension, self.n_filters, 3*self.op_adj.operator.domain.shape[0], self.op_adj.operator.domain.shape[0])
        elif dimension == 2:
            self.dual_block = CnnBlock(dimension, self.n_filters, 3, 1)

    def dual_operation(self, primal:torch.Tensor, dual:torch.Tensor, input_sinogram:torch.Tensor) -> torch.Tensor:
        if self.dimension == 1:
            return dual + self.dual_block( torch.cat([dual, torch.squeeze(self.op(primal)) / self.operator_norm, input_sinogram / self.operator_norm ], dim=1))
        elif self.dimension == 2:
            return dual + self.dual_block( torch.cat([dual, self.op(primal) / self.operator_norm, input_sinogram / self.operator_norm ], dim=1))
        else:
            raise ValueError

    def primal_operation(self, primal:torch.Tensor, dual:torch.Tensor) -> torch.Tensor:
        if self.dimension == 1:
            return primal + self.primal_block(torch.cat([primal, self.op_adj(dual.unsqueeze(1)) / self.operator_norm], dim=1))
        if self.dimension == 2:
            return primal + self.primal_block(torch.cat([primal, self.op_adj(dual) / self.operator_norm], dim=1))
        else:
            raise ValueError

    def forward(self, primal:torch.Tensor, dual:torch.Tensor, input_sinogram:torch.Tensor):
        # dual block
        dual = self.dual_operation(primal, dual, input_sinogram)
        # primal block
        return self.primal_operation(primal, dual), dual

class LearnedPrimalDual(nn.Module):
    def __init__(self, dimension:int, odl_backend:ODLBackend, n_iterations:int, n_filters:int, device:torch.device):
        super(LearnedPrimalDual, self).__init__()
        self.dimension = dimension
        self.pytorch_operator, self.pytorch_operator_adj, self.operator_norm = odl_backend.get_pytorch_operators(device)
        self.operator_domain_shape = self.pytorch_operator.operator.domain.shape # type:ignore
        self.adjoint_domain_shape  = self.pytorch_operator_adj.operator.domain.shape # type:ignore
        self.device = device
        self.n_iterations = n_iterations
        self.iteration_modules = torch.nn.ModuleDict(
            {f'iteration_{i}':Iteration(dimension, self.pytorch_operator, self.pytorch_operator_adj, self.operator_norm, n_filters) for i in range(self.n_iterations)}
            ).to(device)

    def forward(self, input_sinogram:torch.Tensor):

        primal = torch.zeros((input_sinogram.size()[0], 1) + self.operator_domain_shape).to(self.device) #type:ignore
        dual   = torch.zeros((input_sinogram.size()[0], 1) + self.adjoint_domain_shape).to(self.device) #type:ignore

        if self.dimension == 1:
            dual   = torch.squeeze(dual)

        else:
            raise ValueError

        for i in range(self.n_iterations):
            primal, dual = self.iteration_modules[f'iteration_{i}'](primal, dual, input_sinogram)
        return primal[:, 0:1, :]

class Down2D(nn.Module):
    """Down sampling unit of factor 2

        Args:
            input_channels (int): Number of input channels
            output_channels (int): Number of output channels
            filter_size (int): size of the filter of the conv layers, odd integer
    """
    def __init__(self, input_channels:int, output_channels:int, filter_size:int):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(input_channels,  input_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope = 0.1),
            nn.Conv2d(input_channels,  output_channels, filter_size, 1, int((filter_size-1) / 2)),
            nn.LeakyReLU(negative_slope = 0.1),
            nn.Conv2d(output_channels, output_channels, filter_size, 1, int((filter_size - 1) / 2)),
            nn.LeakyReLU(negative_slope = 0.1)
        )
        weights_init(self.down)

    def forward(self, input_tensor:torch.Tensor) -> torch.Tensor:
        """forward function of the Down2D module: input -> output

        Args:
            input_tensor (torch.Tensor): input tensor
        Returns:
            torch.Tensor: output tensor
        """
        return self.down(input_tensor)

class Up2D(nn.Module):
    """Up sampling unit of factor 2

        Args:
            input_channels (int): Number of input channels
            output_channels (int): Number of output channels
    """
    def __init__(self, input_channels:int, output_channels:int):
        super().__init__()
        self.unpooling2d = nn.ConvTranspose2d(input_channels, input_channels, 4, stride = 2, padding = 1)
        self.conv1 = nn.Conv2d(input_channels,  output_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(2 * output_channels, output_channels, 3, stride=1, padding=1)
        self.l_relu = nn.LeakyReLU(negative_slope = 0.1)


    def forward(self, input_tensor:torch.Tensor, skp_connection:torch.Tensor) -> torch.Tensor:
        """forward function of the Up2D module: input -> output

        Args:
            input_tensor (torch.Tensor): input tensor
            skp_connection (torch.Tensor): input from downsampling path

        Returns:
            torch.Tensor: output tensor
        """
        x_0 = self.l_relu(self.unpooling2d(input_tensor))
        x_1 = self.l_relu(self.conv1(x_0))
        return self.l_relu(self.conv2(torch.cat((x_1, skp_connection), 1)))

class Unet2D512(nn.Module):
    """Definition of the 2D unet
    """
    def __init__(self, input_channels:int, output_channels:int, n_filters:int):
        super(Unet2D512, self).__init__()
        # Initialize neural network blocks.
        self.conv1 = nn.Conv2d(input_channels, n_filters, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(n_filters, n_filters, 5, stride=1, padding=2)
        self.down1 = Down2D(n_filters, 2*n_filters, 5)
        self.down2 = Down2D(2*n_filters, 4*n_filters, 3)
        self.down3 = Down2D(4*n_filters, 8*n_filters, 3)
        self.down4 = Down2D(8*n_filters, 16*n_filters, 3)
        self.down5 = Down2D(16*n_filters, 32*n_filters, 3)
        self.down6 = Down2D(32*n_filters, 64*n_filters, 3)
        self.down7 = Down2D(64*n_filters, 64*n_filters, 3)
        self.up1   = Up2D(64*n_filters, 64*n_filters)
        self.up2   = Up2D(64*n_filters, 32*n_filters)
        self.up3   = Up2D(32*n_filters, 16*n_filters)
        self.up4   = Up2D(16*n_filters, 8*n_filters)
        self.up5   = Up2D(8*n_filters, 4*n_filters)
        self.up6   = Up2D(4*n_filters, 2*n_filters)
        self.up7   = Up2D(2*n_filters, n_filters)
        self.conv3 = nn.Conv2d(n_filters +1, 2*output_channels, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(2*output_channels, output_channels, 3, stride=1, padding=1)
        self.l_relu = nn.LeakyReLU(negative_slope=0.1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor :torch.Tensor):
        assert(len(input_tensor.size())==4)
        assert(input_tensor.size()[2]==input_tensor.size()[3]==512)
        s_0  = self.l_relu(self.conv1(input_tensor))
        s_1 = self.l_relu(self.conv2(s_0))
        s_2 = self.down1(s_1)
        s_3 = self.down2(s_2)
        s_4 = self.down3(s_3)
        s_5 = self.down4(s_4)
        s_6 = self.down5(s_5)
        s_7 = self.down6(s_6)
        u_0 = self.down7(s_7)
        u_1 = self.up1(u_0, s_7)
        u_2 = self.up2(u_1, s_6)
        u_3 = self.up3(u_2, s_5)
        u_4 = self.up4(u_3, s_4)
        u_5 = self.up5(u_4, s_3)
        u_6 = self.up6(u_5, s_2)
        u_7 = self.up7(u_6, s_1)
        y_0 = self.l_relu(self.conv3(torch.cat((u_7, input_tensor), 1)))
        y_1 = self.sigmoid(self.conv4(y_0))
        return y_1

if __name__ == '__main__':
    print('models.py')