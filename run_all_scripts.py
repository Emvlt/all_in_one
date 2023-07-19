import pathlib
import subprocess
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scripts_folder_path', default = 'train_scripts_hpc/reconstruction')
    args = parser.parse_args()
    path_to_train_scripts = pathlib.Path(args.scripts_folder_path)
    for path_to_experiment in list(path_to_train_scripts.glob('*')):
        for path_to_metadata in list(path_to_experiment.glob('*')):
            if path_to_metadata.is_file():
                subprocess.run(["sbatch", f"{path_to_metadata}"])