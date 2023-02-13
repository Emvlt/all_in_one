import pathlib

MU_WATER =  0.0192
PHOTONS_PER_PIXEL = 10000
PI = 3.141592653589793

PROJECTPATH  = pathlib.Path(__file__).resolve()
DATASETPATH  = PROJECTPATH.joinpath('datasets')
MODELSPATH   = PROJECTPATH.joinpath('models')
METADATAPATH = PROJECTPATH.joinpath('metadata')

DEFAULT_TIME_STAMP = '50_19_04_Dec_2022'

DEFAULT_METADATA = {
    "training_parameters" : {
        "batch_size":1,
        "learning_rate":1e-3,
        "maximum_steps": 10000
            },
    "architecture_parameters" : {
        "n_iterations":10,
        "n_primal":5,
        "n_dual":5
            },
    "geometry_parameters" : {
        "x_lim": 128,
        "size": 512,
        "angle_start": 0,
        "angle_end": 2*PI,
        "n_angles": 540,
        "src_radius": 512,
        "det_radius": 512,
        "detector_count":736,
        "det_spacing":2.5
            },
    "dataset_parameters" : {
        "dimension":'2D',
        'dataset_name':'Mayo',
        'modality': 'sinograms'
        },
    "device": "cuda:0"
    }