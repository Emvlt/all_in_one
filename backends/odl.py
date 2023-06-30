from typing import Dict

import odl
import odl.contrib.torch as odl_torch
import torch
import numpy as np

class ODLBackend():
    def __init__(self) -> None:
        self.space = None
        self.angle_partition = None
        self.detector_partition = None
        self.geometry = None
        self.operator = None
        self.pytorch_operator = None
        self.pytorch_operator_adjoint = None
        self.pytorch_operator_norm:torch.Tensor = torch.Tensor()
        self.initialised = False

    def initialise_space_from_dict(self, space_dict:Dict) -> None:
        self.space = odl.uniform_discr(
            min_pt = space_dict['min_pt'],
            max_pt = space_dict['max_pt'],
            shape  = space_dict['shape'],
            dtype = space_dict['dtype']
        )
        self.space_dict = space_dict

    def initialise_angle_partition_from_dict(self, angle_partition_dict:Dict)-> None:
        self.angle_partition = odl.uniform_partition(
            min_pt = angle_partition_dict['min_pt'],
            max_pt = angle_partition_dict['max_pt'],
            shape  = angle_partition_dict['shape']
        )
        self.angle_partition_dict = angle_partition_dict

    def initialise_detector_partition_from_dict(self, detector_partition_dict:Dict)-> None:
        self.detector_partition = odl.uniform_partition(
            min_pt = detector_partition_dict['min_pt'],
            max_pt = detector_partition_dict['max_pt'],
            shape  = detector_partition_dict['shape']
        )
        self.detector_partition_dict = detector_partition_dict

    def initialise_geometry_from_dict(self, geometry_dict:Dict) -> None:
        assert self.angle_partition is not None, 'Angle partition not initialised, consider running odl_backend.initialise_angle_partition_from_dict'
        assert self.detector_partition is not None, 'Angle partition not initialised, consider running odl_backend.initialise_detector_partition_from_dict'
        if geometry_dict['beam_geometry'] == 'fan_beam':
            self.geometry = odl.tomo.FanBeamGeometry(
                apart = self.angle_partition,
                dpart = self.detector_partition,
                src_radius=geometry_dict['src_radius'],
                det_radius=geometry_dict['det_radius']
                )
            self.geometry_dict = geometry_dict
        else:
            raise NotImplementedError (f"Beam geometry {geometry_dict['beam_geometry']} not implemented")

    def set_operator(self) -> None:
        assert self.space is not None, 'space attribute not initialised, consider running odl_backend.initialise_space_from_dict'
        assert self.geometry is not None, 'space attribute not initialised, consider running odl_backend.set_geometry'
        self.operator = odl.tomo.RayTransform(self.space, self.geometry)

    def set_pytorch_operator(self):
        assert self.operator is not None, 'operator attribute not initialised, consider running odl_backend.set_operator'
        self.pytorch_operator = odl_torch.OperatorModule(self.operator)

    def set_pytorch_operator_adjoint(self):
        assert self.operator is not None, 'operator attribute not initialised, consider running odl_backend.set_operator'
        self.pytorch_operator_adjoint = odl_torch.OperatorModule(self.operator.adjoint)

    def set_pytorch_operator_norm(self):
        assert self.operator is not None, 'operator attribute not initialised, consider running odl_backend.set_operator'
        self.pytorch_operator_norm = torch.as_tensor(self.operator.norm(estimate=True))

    def set_pytorch_operators(self):
        self.set_pytorch_operator()
        self.set_pytorch_operator_adjoint()
        self.set_pytorch_operator_norm()

    def initialise_odl_backend_from_metadata_dict(self, metadata_dict:Dict):
        self.initialise_angle_partition_from_dict(metadata_dict['angle_partition_dict'])
        self.initialise_detector_partition_from_dict(metadata_dict['detector_partition_dict'])
        self.initialise_space_from_dict(metadata_dict['space_dict'])
        self.initialise_geometry_from_dict(metadata_dict['geometry_dict'])

        self.set_operator()

        self.set_pytorch_operators()

        self.initialised = True

    def get_pytorch_operators(self, device:torch.device):
        assert self.initialised, 'ODL backend not initialised, consider running odl_backend.initialise_odl_backend_from_metadata_dict'
        return self.pytorch_operator.to(device), self.pytorch_operator_adjoint.to(device), self.pytorch_operator_norm.to(device) # type:ignore

    def get_sinogram(self, sample:torch.Tensor) -> torch.Tensor:
        assert self.initialised, 'ODL backend not initialised, consider running odl_backend.initialise_odl_backend_from_metadata_dict'
        return self.pytorch_operator(sample) / self.pytorch_operator_norm # type:ignore

    def get_reconstruction(self, sinogram:torch.Tensor) -> torch.Tensor:
        assert self.initialised, 'ODL backend not initialised, consider running odl_backend.initialise_odl_backend_from_metadata_dict'
        return self.pytorch_operator_adjoint(sinogram)  # type:ignore

    def get_filtered_backprojection(self, sinogram:np.ndarray, filter_name:str) -> np.ndarray:
        assert self.initialised, 'ODL backend not initialised, consider running odl_backend.initialise_odl_backend_from_metadata_dict'
        fbp = odl.tomo.fbp_op(self.operator, filter_type=filter_name, frequency_scaling=0.8)
        return fbp(sinogram)

    def get_filtered_backprojection_operator(self, filter_name:str):
        assert self.initialised, 'ODL backend not initialised, consider running odl_backend.initialise_odl_backend_from_metadata_dict'
        return odl.tomo.fbp_op(self.operator, filter_type=filter_name, frequency_scaling=0.8)