import odl
import numpy as np

readings_loc = r'C:\Users\hx21262\Downloads\TEMP_XNAT\100_20200414\resources\RAW\files\66862.Anonymous.CT..601.RAW.20200414.110431.963978.2020.07.03.21.11.10.086000.1309236364\readings.npy'

n_readings = 39868
voxels_x = 128
voxels_y = 128
voxels_z = 128
voxels_min_pt = [-20, -20, 0]
voxels_max_pt = [20, 20, 40]

reco_space = odl.uniform_discr(
    min_pt=voxels_min_pt, max_pt=voxels_max_pt, shape=[voxels_x,voxels_y,voxels_z],
    dtype='float32')

# Make a helical cone beam geometry with flat detector
angles_partition_min = 0
angles_partition_max = 2*np.pi*0.067864004196156*n_readings/360
# Angles: uniformly spaced, n = 2000, min = 0, max = 8 * 2 * pi
angle_partition = odl.uniform_partition(angles_partition_min,angles_partition_max, n_readings)

row_width = 736
n_rows = 32
detector_min = [-50, -3]
detector_max = [50, 3]
# Detector: uniformly sampled, n = (512, 64), min = (-50, -3), max = (50, 3)
detector_partition = odl.uniform_partition(detector_min, detector_max, [row_width, n_rows])
# Spiral has a pitch of 5, we run 8 rounds (due to max angle = 8 * 2 * pi)
src_radius = 100
det_radius = 100
pitch = 5.0
geometry = odl.tomo.ConeBeamGeometry (
    angle_partition, detector_partition, src_radius=src_radius, det_radius=det_radius, pitch=pitch)

# Ray transform (= forward projection).
ray_trafo = odl.tomo.RayTransform(reco_space, geometry)

# Create a discrete Shepp-Logan phantom (modified version)
phantom = odl.phantom.shepp_logan(reco_space, modified=True)

# Create projection data by calling the ray transform on the phantom
proj_data = np.load(readings_loc)

# Back-projection can be done by simply calling the adjoint operator on the
# projection data (or any element in the projection space).
backproj = ray_trafo.adjoint(proj_data)

# Shows a slice of the phantom, projections, and reconstruction
phantom.show(coords=[None, None, 20], title='Phantom, Middle Z Slice')
proj_data.show(coords=[2 * np.pi, None, None],
               title='Projection After Exactly One Turn')
proj_data.show(coords=[None, None, 0], title='Sinogram, Middle Slice')
backproj.show(coords=[None, None, 20], title='Back-projection, Middle Z Slice',
              force_show=True)
