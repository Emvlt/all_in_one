from typing import Dict
import pathlib
import torch

import odl
import odl.contrib.torch as odl_torch

from constants import DATASETPATH, DEFAULT_METADATA

def generate_geometry(device:torch.device, geometry_parameters:Dict):
    space = odl.uniform_discr([-geometry_parameters['x_lim']]*2, [geometry_parameters['x_lim']]*2, [geometry_parameters['size']]*2, dtype='float32')
    angle_partition = odl.uniform_partition(geometry_parameters['angle_start'], geometry_parameters['angle_end'], geometry_parameters['n_angles'])
    detector_partition = odl.uniform_partition(-360, 360, geometry_parameters['detector_count'])
    geometry = odl.tomo.FanBeamGeometry(angle_partition, detector_partition, src_radius=geometry_parameters['src_radius'], det_radius=geometry_parameters['det_radius'])
    operator = odl.tomo.RayTransform(space, geometry)
    # transform the operator and it's adjoint into pytorch modules
    pytorch_operator = odl_torch.OperatorModule(operator)
    pytorch_operator_adjoint = odl_torch.OperatorModule(operator.adjoint)
    # after each application of the operator, divide the result by it's norm
    # for numerical stability
    operator_norm = torch.as_tensor(operator.norm(estimate=True))
    return pytorch_operator.to(device), pytorch_operator_adjoint.to(device), operator_norm.to(device)

def sample_phantom(geometry_parameters:Dict, input_path:pathlib.Path, output_path:pathlib.Path):
    pytorch_operator, pytorch_operator_adjoint, operator_norm = generate_geometry(geometry_parameters)
    try:
        sample:torch.Tensor = torch.load(str(input_path))
    except:
        raise FileNotFoundError(f'There is no file at {input_path}')
    assert(len(sample.size())==4)
    sinogram :torch.Tensor= pytorch_operator(sample) / operator_norm
    torch.save(sinogram, output_path)
    import matplotlib.pyplot as plt
    plt.matshow(sinogram[0,0].detach())
    plt.show()

if __name__ == '__main__':
    metadata = DEFAULT_METADATA
    sample_phantom(metadata['geometry_parameters'], DATASETPATH.joinpath('2D/Custom/test_phantom.pt'), DATASETPATH.joinpath('2D/Custom/test_sinopgram.pt'))
