import argparse
import pathlib
import copy

import json

PIPELINE_FOLDERS = {
    "reconstruction": [
        "6_percent_measurements",
        "25_percent_measurements",
        "100_percent_measurements",
    ],
    "segmentation": [
        "from_input_images",
        "6_percent_measurements",
        "25_percent_measurements",
        "100_percent_measurements",
    ],
}
FOLDER_NAME_TO_MEASUREMENTS = {
    '6_percent_measurements':64,
    '25_percent_measurements':256,
    '100_percent_measurements':1024
    }

metadata_template = {
    "data_feeding_dict":{
        "dataset_name":"LIDC-IDRI",
        "train":True,
        "training_proportion":0.8,
        "is_subset":False,
        "shuffle":True,
        "batch_size":8,
        "num_workers":2
    },

    "training_dict":{
        "learning_rate":1e-4,
        "n_epochs":1,
        "dose":1,
        "dual_loss_weighting":0,
        "reconstruction_loss":"MSE",
        "sinogram_loss":"L1"
    },

    "architecture_dict":{
        "reconstruction":{
            "name":"lpd",
            "train":True,
            "primal_dict":{
                "dimension":2,
                "name":"Unet",
                "n_layers":1,
                "n_filters":8
            },
            "dual_dict":{
                "dimension":2,
                "name":"Unet",
                "n_layers":1,
                "n_filters":8
            },
            "n_iterations":5,
            "fourier_filtering_dict":{
                "is_filter":False
            },
            "load_path":"",
            "device_name":"cuda:0"
        }
    },

    "scan_parameter_dict":{
        "space_dict":{
            "min_pt":[-192,-192],
            "max_pt":[192,192],
            "shape":[512,512],
            "dtype":"float32"
        },
        "angle_partition_dict":{
            "min_pt":0,
            "max_pt":6.28318530718,
            "shape":64
        },
        "detector_partition_dict":{
            "min_pt":-473,
            "max_pt":473,
            "shape":736
        },
        "geometry_dict":{
            "src_radius":787,
            "det_radius":298,
            "beam_geometry":"fan_beam"
        }

    }

}

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", required=False, default='holly-b')
    parser.add_argument("--pipeline", required=False, default='reconstruction')
    args = parser.parse_args()

    print(f'Writing metadata for pipeline {args.pipeline}')
    for folder_name in PIPELINE_FOLDERS[args.pipeline]:
        print(f'Processing folder {folder_name}')
        folder_save_path = pathlib.Path(f'metadata_folder/{args.pipeline}/{folder_name}')
        folder_save_path.mkdir(parents=True, exist_ok=True)
        metadata_file = copy.deepcopy(metadata_template)
        ## Modify n measurements
        metadata_file["scan_parameter_dict"]["angle_partition_dict"]["shape"] = FOLDER_NAME_TO_MEASUREMENTS[folder_name]
        for dual_loss_weighting in [0,0.5]:
            for sinogram_loss in ['L1', 'MSE']:
                for dual_dimension in [1,2]:
                    for n_iterations in [1,5]:
                        metadata_file["training_dict"]["dual_loss_weighting"] = dual_loss_weighting
                        metadata_file["training_dict"]["sinogram_loss"] = sinogram_loss
                        metadata_file["architecture_dict"]["reconstruction"]["dual_dict"]["dimension"] = dual_dimension
                        metadata_file["architecture_dict"]["reconstruction"]["n_iterations"] = n_iterations
                        save_path = folder_save_path / f'full_dataset_{dual_dimension}d_{n_iterations}it_sinogram_{sinogram_loss}_{dual_loss_weighting}_loss.json'
                        with(open(save_path, 'w')) as out_f:
                            json.dump(metadata_file, out_f, indent = 4)