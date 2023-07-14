import pathlib

import torch.nn as nn
import torch
from backends.odl import ODLBackend


def load_network(
    models_load_path: pathlib.Path,
    network: torch.nn.Module,
    load_path,
    indent_level=None,
):
    if isinstance(load_path, str):
        load_path = pathlib.Path(load_path)
    elif isinstance(load_path, pathlib.Path):
        pass
    else:
        raise TypeError(
            f"Wrong type for load_path argument {load_path}, must be str or pathlib.Path"
        )

    model_load_path = models_load_path / load_path

    offset = 0
    if indent_level is not None:
        offset = indent_level

    if model_load_path.is_file():
        print(
            "\t" * offset + f"Loading model state_dict from {model_load_path}"
        )  # type:ignore
        network.load_state_dict(torch.load(model_load_path))
    else:
        print(
            "\t" * offset + f"No file found at {model_load_path}, no initialisation"
        )  # type:ignore
    return network


def weights_init(module: nn.Module):
    if isinstance(module, nn.Conv1d):
        shape = module.weight.shape
        lim = (6 / (shape[0] + shape[1]) / shape[2]) ** 0.5
        module.weight.data.uniform_(-lim, lim)
        module.bias.data.fill_(0)  # type:ignore

    elif isinstance(module, nn.Conv2d):
        shape = module.weight.shape
        lim = (6 / (shape[0] + shape[1]) / shape[2] / shape[3]) ** 0.5
        module.weight.data.uniform_(-lim, lim)
        module.bias.data.fill_(0)  # type:ignore


class FourierTransformInceptionLayer(nn.Module):
    def __init__(
        self,
        dimension: int,
        n_channels_input: int,
        n_channels_output: int,
        n_filters: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super(FourierTransformInceptionLayer, self).__init__()
        if dimension == 1:
            self.conv1 = nn.Conv1d(
                n_channels_input, n_filters, 1, padding=0, dtype=dtype, device=device
            )
            self.conv3 = nn.Conv1d(
                n_channels_input, n_filters, 3, padding=1, dtype=dtype, device=device
            )
            self.conv5 = nn.Conv1d(
                n_channels_input, n_filters, 5, padding=2, dtype=dtype, device=device
            )
            self.conv7 = nn.Conv1d(
                n_channels_input, n_filters, 7, padding=3, dtype=dtype, device=device
            )
            self.collection_filter = nn.Conv1d(
                4 * n_filters,
                n_channels_output,
                7,
                padding=3,
                dtype=dtype,
                device=device,
            )

        elif dimension == 2:
            self.conv1 = nn.Conv2d(
                n_channels_input,
                n_filters,
                (1, 1),
                padding=(0, 0),
                dtype=dtype,
                device=device,
            )
            self.conv3 = nn.Conv2d(
                n_channels_input,
                n_filters,
                (1, 3),
                padding=(0, 1),
                dtype=dtype,
                device=device,
            )
            self.conv5 = nn.Conv2d(
                n_channels_input,
                n_filters,
                (1, 5),
                padding=(0, 2),
                dtype=dtype,
                device=device,
            )
            self.conv7 = nn.Conv2d(
                n_channels_input,
                n_filters,
                (1, 7),
                padding=(0, 3),
                dtype=dtype,
                device=device,
            )
            self.collection_filter = nn.Conv2d(
                4 * n_filters,
                n_channels_output,
                7,
                padding=3,
                dtype=dtype,
                device=device,
            )

        self.filtering = nn.Sequential(self.collection_filter)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.filtering(
            torch.cat(
                [
                    self.conv1(input_tensor),
                    self.conv3(input_tensor),
                    self.conv5(input_tensor),
                    self.conv7(input_tensor),
                ],
                dim=1,
            )
        )


class AverageModule(nn.Module):
    def __init__(self, n_measurements: int, n_filters: int, device: torch.device):
        super(AverageModule, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=n_measurements,
            out_channels=n_measurements,
            kernel_size=(n_filters, 1),
            padding=(0, 0),
            dtype=torch.cfloat,
            device=device,
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=n_measurements,
            out_channels=n_measurements,
            kernel_size=(n_filters, 3),
            padding=(0, 1),
            dtype=torch.cfloat,
            device=device,
        )
        self.conv3 = torch.nn.Conv2d(
            in_channels=n_measurements,
            out_channels=n_measurements,
            kernel_size=(n_filters, 5),
            padding=(0, 2),
            dtype=torch.cfloat,
            device=device,
        )
        self.conv4 = torch.nn.Conv2d(
            in_channels=n_measurements,
            out_channels=n_measurements,
            kernel_size=(n_filters, 7),
            padding=(0, 3),
            dtype=torch.cfloat,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.squeeze(
            self.conv1(x) + self.conv2(x) + self.conv3(x) + self.conv4(x)
        )


class InceptionLayer(nn.Module):
    def __init__(
        self,
        dimension: int,
        n_channels_input: int,
        n_channels_output: int,
        n_filters: int,
        dtype=torch.float,
    ) -> None:
        super(InceptionLayer, self).__init__()
        if dimension == 1:
            self.conv3 = nn.Conv1d(
                n_channels_input, n_filters, 3, padding=1, dtype=dtype
            )
            self.conv5 = nn.Conv1d(
                n_channels_input, n_filters, 5, padding=2, dtype=dtype
            )
            self.conv7 = nn.Conv1d(
                n_channels_input, n_filters, 7, padding=3, dtype=dtype
            )
            self.collection_filter = nn.Conv1d(
                3 * n_filters, n_channels_output, 7, padding=3, dtype=dtype
            )

        elif dimension == 2:
            self.conv3 = nn.Conv2d(
                n_channels_input, n_filters, 3, padding=1, dtype=dtype
            )
            self.conv5 = nn.Conv2d(
                n_channels_input, n_filters, 5, padding=2, dtype=dtype
            )
            self.conv7 = nn.Conv2d(
                n_channels_input, n_filters, 7, padding=3, dtype=dtype
            )
            self.collection_filter = nn.Conv2d(
                3 * n_filters, n_channels_output, 7, padding=3, dtype=dtype
            )

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.filtering = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.1),
            self.collection_filter,
            nn.LeakyReLU(negative_slope=0.1),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.filtering(
            torch.cat(
                [
                    self.conv3(input_tensor),
                    self.conv5(input_tensor),
                    self.conv7(input_tensor),
                ],
                dim=1,
            )
        )


class CnnBlock(nn.Module):
    def __init__(
        self,
        dimension: int,
        n_filters: int,
        n_channels_input: int,
        n_channels_output: int,
    ) -> None:
        super(CnnBlock, self).__init__()
        if dimension == 1:
            self.block = nn.Sequential(
                nn.Conv1d(n_channels_input, n_filters, 3, padding=1),
                nn.PReLU(num_parameters=n_filters, init=0.0),
                nn.Conv1d(n_filters, n_filters, 3, padding=1),
                nn.PReLU(num_parameters=n_filters, init=0.0),
                nn.Conv1d(n_filters, n_channels_output, 3, padding=1),
            )
        elif dimension == 2:
            self.block = nn.Sequential(
                nn.Conv2d(n_channels_input, n_filters, 3, padding=1),
                nn.PReLU(num_parameters=n_filters, init=0.0),
                nn.Conv2d(n_filters, n_filters, 3, padding=1),
                nn.PReLU(num_parameters=n_filters, init=0.0),
                nn.Conv2d(n_filters, n_channels_output, 3, padding=1),
            )

        weights_init(self.block)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.block(input_tensor)


class DownModule(nn.Module):
    def __init__(
        self,
        dimension: int,
        n_channels_input: int,
        n_channels_output: int,
        n_filters: int,
    ):
        super().__init__()
        if dimension == 1:
            self.down = nn.Sequential(
                nn.Conv1d(n_channels_input, n_channels_output, 4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv1d(n_channels_output, n_channels_output, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv1d(n_channels_output, n_channels_output, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1),
            )
        elif dimension == 2:
            self.down = nn.Sequential(
                nn.Conv2d(n_channels_input, n_channels_output, 4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv2d(n_channels_output, n_channels_output, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv2d(n_channels_output, n_channels_output, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1),
            )
        weights_init(self.down)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.down(input_tensor)


class UpModule(nn.Module):
    def __init__(
        self,
        dimension: int,
        n_channels_input: int,
        n_channels_output: int,
        n_filters: int,
    ):
        super().__init__()
        if dimension == 1:
            self.up = nn.ConvTranspose1d(
                n_channels_input, n_channels_input, 4, stride=2, padding=1
            )
            self.conv1 = nn.Conv1d(n_channels_input, n_channels_output, 5, 1, 2)
            self.conv2 = nn.Conv1d(2 * n_channels_output, n_channels_output, 5, 1, 2)
        elif dimension == 2:
            self.up = nn.ConvTranspose2d(
                n_channels_input, n_channels_input, 4, stride=2, padding=1
            )
            self.conv1 = nn.Conv2d(n_channels_input, n_channels_output, 5, 1, 2)
            self.conv2 = nn.Conv2d(2 * n_channels_output, n_channels_output, 5, 1, 2)
        """ self.conv1 = InceptionLayer(dimension, n_channels_input,  n_channels_output, n_filters)
        self.conv2 = InceptionLayer(dimension, 2*n_channels_output,  n_channels_output, n_filters)"""
        self.l_relu = nn.LeakyReLU(negative_slope=0.1)

    def forward(
        self, input_tensor: torch.Tensor, skp_connection: torch.Tensor
    ) -> torch.Tensor:
        x_0 = self.l_relu(self.up(input_tensor))
        x_1 = self.l_relu(self.conv1(x_0))
        return self.l_relu(self.conv2(torch.cat((x_1, skp_connection), 1)))


class Unet(nn.Module):
    def __init__(
        self,
        dimension: int,
        n_channels_input: int,
        n_channels_output: int,
        n_filters: int,
        regression=True,
    ):
        super(Unet, self).__init__()
        # Initialize neural network blocks.
        self.conv1 = InceptionLayer(dimension, n_channels_input, n_filters, n_filters)
        self.conv2 = InceptionLayer(dimension, n_filters, n_filters, n_filters)
        self.down1 = DownModule(dimension, n_filters, 2 * n_filters, n_filters)
        self.down2 = DownModule(dimension, 2 * n_filters, 4 * n_filters, n_filters)
        self.down3 = DownModule(dimension, 4 * n_filters, 8 * n_filters, n_filters)
        self.down4 = DownModule(dimension, 8 * n_filters, 16 * n_filters, n_filters)
        self.down5 = DownModule(dimension, 16 * n_filters, 32 * n_filters, n_filters)
        self.up1 = UpModule(dimension, 32 * n_filters, 16 * n_filters, n_filters)
        self.up2 = UpModule(dimension, 16 * n_filters, 8 * n_filters, n_filters)
        self.up3 = UpModule(dimension, 8 * n_filters, 4 * n_filters, n_filters)
        self.up4 = UpModule(dimension, 4 * n_filters, 2 * n_filters, n_filters)
        self.up5 = UpModule(dimension, 2 * n_filters, n_filters, n_filters)
        self.conv3 = InceptionLayer(
            dimension, n_filters + n_channels_input, 2 * n_channels_output, n_filters
        )
        self.conv4 = InceptionLayer(
            dimension, 2 * n_channels_output, n_channels_output, n_filters
        )
        self.l_relu = nn.LeakyReLU(negative_slope=0.1)
        if regression:
            self.last_layer = self.l_relu
        else:
            self.last_layer = nn.Sigmoid()

    def forward(self, input_tensor: torch.Tensor):
        s_0 = self.l_relu(self.conv1(input_tensor))
        s_1 = self.l_relu(self.conv2(s_0))
        s_2 = self.down1(s_1)
        s_3 = self.down2(s_2)
        s_4 = self.down3(s_3)
        s_5 = self.down4(s_4)
        u_0 = self.down5(s_5)
        u_1 = self.up1(u_0, s_5)
        u_2 = self.up2(u_1, s_4)
        u_3 = self.up3(u_2, s_3)
        u_4 = self.up4(u_3, s_2)
        u_5 = self.up5(u_4, s_1)
        return self.last_layer(
            self.conv4(self.l_relu(self.conv3(torch.cat((u_5, input_tensor), 1))))
        )


class FilterModule(nn.Module):
    def __init__(
        self,
        dimension: int,
        detector_size: int,
        filter_name: str,
        std: float,
        device: torch.device,
        training_mode=False,
    ):
        super(FilterModule, self).__init__()
        if dimension == 1:
            linspace = torch.linspace(
                -0.5, 0.5, detector_size, dtype=torch.cfloat, device=device
            )
        else:
            raise NotImplementedError(
                "Filter Module not implemented for dimension != 1"
            )
        if filter_name == "custom":
            linspace.imag = (
                0.5 - torch.exp(-torch.square(linspace.real) / (2 * std**2)) * 0.5
            )
            linspace.real = (
                0.5 - torch.exp(-torch.square(linspace.real) / (2 * std**2)) * 0.5
            )
        elif filter_name == "ramp":
            linspace.imag = torch.abs(linspace.imag)
            linspace.real = torch.abs(linspace.real)

        self.weight = torch.nn.Parameter(linspace)

        self.weight.requires_grad = training_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mul(x, self.weight)


class FourierFilteringModule(nn.Module):
    def __init__(
        self,
        dimension: int,
        n_measurements: int,
        detector_size: int,
        device: torch.device,
        filter_name: str,
        training_mode: bool,
    ):
        super(FourierFilteringModule, self).__init__()
        if dimension != 1:
            raise NotImplementedError(
                "Fourier Filtering module not implemented for dimension !=1"
            )

        self.dimension = dimension
        self.n_measurements = n_measurements
        self.detector_size = detector_size

        self.filter = FilterModule(
            dimension, self.detector_size, filter_name, 0.1, device, training_mode
        )

    def forward(self, sinogram: torch.Tensor) -> torch.Tensor:
        # sinogram size : [B_s, n_measurements, Det_size]
        fourier_transform: torch.Tensor = torch.fft.fft(sinogram)
        centered_fourier_transform = torch.fft.fftshift(fourier_transform)
        # fourier_transform size : [B_s, n_measurements, Det_size]
        filtered = self.filter(centered_fourier_transform)
        return torch.real(torch.fft.ifft(torch.fft.ifftshift(filtered)))


class Iteration(nn.Module):
    def __init__(
        self,
        dimension: int,
        odl_backend: ODLBackend,
        n_measurements: int,
        detector_size: int,
        n_primal: int,
        n_dual: int,
        n_filters_primal: int,
        n_filters_dual: int,
        fourier_filtering: bool,
        device: torch.device,
        filter_name="",
        training_mode=False,
    ):
        super(Iteration, self).__init__()
        self.dimension = dimension
        (
            pytorch_operator,
            pytorch_operator_adj,
            operator_norm,
        ) = odl_backend.get_pytorch_operators(device)
        self.op = pytorch_operator
        self.op_adj = pytorch_operator_adj
        self.operator_norm = operator_norm
        self.n_measurements = n_measurements
        self.fourier_filtering = fourier_filtering

        self.primal_block = Unet(2, n_primal + 1, n_primal, n_filters_primal).to(device)
        self.dual_block = Unet(
            dimension,
            (n_dual + 2) * self.n_measurements,
            n_dual * self.n_measurements,
            n_filters_dual,
        ).to(device)

        if self.fourier_filtering:
            self.fourier_sinogram_filtering_module = FourierFilteringModule(
                dimension,
                self.n_measurements,
                detector_size,
                device,
                filter_name,
                training_mode,
            )

    def dual_operation(
        self, primal: torch.Tensor, dual: torch.Tensor, input_sinogram: torch.Tensor
    ) -> torch.Tensor:
        if self.dimension == 1:
            """
            ### Forward and reduce dimension
            x = self.op(primal[:, 0:1, ...]).squeeze()
            ### Concat of dual, x and input sinogram
            x = torch.cat([dual, x / self.operator_norm, input_sinogram / self.operator_norm ], dim=1)
            ### Forward through NN
            x = self.dual_block(x)
            ### Dual addition
            x += dual
            """
            return dual + self.dual_block(
                torch.cat(
                    [
                        dual,
                        self.op(primal[:, 0:1, ...]).squeeze(dim=1)
                        / self.operator_norm,
                        input_sinogram / self.operator_norm,
                    ],
                    dim=1,
                )
            )
        elif self.dimension == 2:
            return dual + self.dual_block(
                torch.cat(
                    [
                        dual,
                        self.op(primal[:, 0:1, ...]) / self.operator_norm,
                        input_sinogram / self.operator_norm,
                    ],
                    dim=1,
                )
            )
        else:
            raise ValueError

    def primal_operation(
        self, primal: torch.Tensor, dual: torch.Tensor
    ) -> torch.Tensor:
        if self.dimension == 1:
            """
            ### Get first dual channel, unsqueeze and back-projection
            x = self.op_adj(dual.unsqueeze(1)) / self.operator_norm
            ### Concat of primal and x
            x = torch.cat([primal, x / self.operator_norm], dim=1)
            ### Forward through NN
            x = self.primal_block(x)
            ### Primal addition
            x += primal
            """
            return primal + self.primal_block(
                torch.cat(
                    [
                        primal,
                        self.op_adj(dual[:, : self.n_measurements].unsqueeze(dim=1))
                        / self.operator_norm,
                    ],
                    dim=1,
                )
            )
        elif self.dimension == 2:
            return primal + self.primal_block(
                torch.cat(
                    [primal, self.op_adj(dual[:, 0:1, ...]) / self.operator_norm], dim=1
                )
            )
        else:
            raise ValueError

    def forward(
        self, primal: torch.Tensor, dual: torch.Tensor, input_sinogram: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # dual block
        dual = self.dual_operation(primal, dual, input_sinogram)
        if self.fourier_filtering:
            dual = self.fourier_sinogram_filtering_module(dual)
        # primal block
        return self.primal_operation(primal, dual), dual


class LearnedPrimalDual(nn.Module):
    def __init__(
        self,
        dimension: int,
        odl_backend: ODLBackend,
        n_primal: int,
        n_dual: int,
        n_iterations: int,
        n_filters_primal: int,
        n_filters_dual: int,
        fourier_filtering: bool,
        device: torch.device,
        fourier_filter_name="",
        training_mode=False,
    ):
        super(LearnedPrimalDual, self).__init__()
        self.dimension = dimension
        self.odl_backend = odl_backend
        self.operator_domain_shape = self.odl_backend.space_dict["shape"]  # type:ignore
        if dimension == 1:
            self.n_measurements = self.odl_backend.angle_partition_dict["shape"]
        elif dimension == 2:
            self.n_measurements = 1
        else:
            raise ValueError
        self.detector_size = self.odl_backend.detector_partition_dict["shape"]
        self.adjoint_domain_shape = [self.n_measurements, self.detector_size]

        self.device = device
        self.n_primal = n_primal
        self.n_dual = n_dual
        self.n_iterations = n_iterations
        self.iteration_modules = torch.nn.ModuleDict(
            {
                f"iteration_{i}": Iteration(
                    dimension,
                    odl_backend,
                    self.n_measurements,
                    self.detector_size,
                    n_primal,
                    n_dual,
                    n_filters_primal,
                    n_filters_dual,
                    fourier_filtering,
                    device,
                    fourier_filter_name,
                    training_mode,
                )
                for i in range(self.n_iterations)
            }
        )

    def forward(
        self, input_sinogram: torch.Tensor, just_infer=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        primal = torch.zeros(
            [input_sinogram.size()[0], self.n_primal] + self.operator_domain_shape
        ).to(
            self.device
        )  # type:ignore

        if self.dimension == 1:
            dual = torch.zeros(
                input_sinogram.size()[0],
                self.n_dual * self.adjoint_domain_shape[0],
                self.adjoint_domain_shape[1],
            ).to(
                self.device
            )  # type:ignore

        elif self.dimension == 2:
            dual = torch.zeros(
                [input_sinogram.size()[0], self.n_dual] + self.adjoint_domain_shape
            ).to(
                self.device
            )  # type:ignore

        else:
            raise ValueError

        for i in range(self.n_iterations):
            primal, dual = self.iteration_modules[f"iteration_{i}"](
                primal, dual, input_sinogram
            )

        if just_infer:
            return primal[:, 0:1]  # type:ignore
        else:
            if self.dimension == 1:
                return primal[:, 0:1], dual[:, : self.n_measurements]
            elif self.dimension == 2:
                return primal[:, 0:1], dual[:, :1]
            else:
                raise ValueError
