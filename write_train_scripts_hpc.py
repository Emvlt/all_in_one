import argparse
import pathlib
import csv
from typing import Dict
import subprocess

from utils import load_json

DEFAULT_WALLCLOCK_TIME = '02:00:00'

WALLCLOCK_TIME_DICT = {
    'reconstruction/100_percent_measurements/lpd_unet_1_iteration':'12:00:00',
    'reconstruction/100_percent_measurements/1d_lpd_unet_1_iteration':'12:00:00'
}

def compute_n_gpus(metadata_dict):
    device_list = []
    architecture_dict:Dict = metadata_dict['architecture_dict']
    for network_dict in architecture_dict.values():
        device_name = network_dict['device_name']
        if device_name not in device_list:
            device_list.append(device_name)
    return len(device_list)

def write_train_script(metadata_path:pathlib.Path):
    print(f"Writing training Script for file at {metadata_path}")
    metadata_dict = load_json(metadata_path)
    pipeline = metadata_path.parent.parent.stem
    experiment_folder_name = metadata_path.parent.stem
    run_name = metadata_path.stem
    print('\t' + f'Pipeline: {pipeline}; Experiment folder name: {experiment_folder_name}; run name: {run_name}')
    train_script_path = pathlib.Path(f'train_scripts_hpc/{pipeline}/{experiment_folder_name}/{run_name}.sh')
    if train_script_path.is_file():
        print(f'{train_script_path} already exists, passing')
        return
    with open(train_script_path, 'w', newline='\n') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter='\n', quotechar=None, escapechar='\t')
        ### Write Shebang
        csv_writer.writerow(['#!/bin/bash'])
        ### Write Header
        csv_writer.writerow([f'#SBATCH -J {run_name}'])
        csv_writer.writerow([f'#SBATCH -A SCHONLIEB-SL3-GPU'])
        csv_writer.writerow([f'#SBATCH --nodes=1'])
        csv_writer.writerow([f'#SBATCH --ntasks=1'])
        csv_writer.writerow([f'#SBATCH --gres=gpu:{compute_n_gpus(metadata_dict)}'])
        if f'{pipeline}/{experiment_folder_name}/{run_name}' in WALLCLOCK_TIME_DICT:
            wallclock_time = WALLCLOCK_TIME_DICT[f'{pipeline}/{experiment_folder_name}/{run_name}']
        else:
            wallclock_time = DEFAULT_WALLCLOCK_TIME
        csv_writer.writerow([f'#SBATCH --time={wallclock_time}'])
        csv_writer.writerow([f'#SBATCH --mail-type=NONE'])
        csv_writer.writerow([f'#SBATCH --no-requeue'])
        csv_writer.writerow([f'#SBATCH -p ampere'])

        csv_writer.writerow([f'numnodes=$SLURM_JOB_NUM_NODES'])
        csv_writer.writerow([f'numtasks=$SLURM_NTASKS'])
        csv_writer.writerow([r'mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE"' + r" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')"])

        csv_writer.writerow(['. /etc/profile.d/modules.sh'])
        csv_writer.writerow(['module purge'])
        csv_writer.writerow(['module load rhel8/default-amp'])

        csv_writer.writerow(['. /etc/profile.d/modules.sh'])
        csv_writer.writerow(['module load miniconda/3'])
        csv_writer.writerow(['source /home/ev373/.bashrc'])
        csv_writer.writerow(['conda activate all_in_one'])

        csv_writer.writerow([f'application="/home/ev373/.conda/envs/all_in_one/bin/python"'])
        csv_writer.writerow([f'options="experiences.py --platform hpc --metadata_path {metadata_path}"'])
        csv_writer.writerow([f'workdir="/home/ev373/work/all_in_one"'])

        csv_writer.writerow(['export OMP_NUM_THREADS=1'])
        csv_writer.writerow(['np=$[${numnodes}*${mpi_tasks_per_node}]'])
        csv_writer.writerow(['CMD="$application $options"'])

        csv_writer.writerow(['cd $workdir'])
        csv_writer.writerow(['echo -e "Changed directory to `pwd`.\\n"'])

        csv_writer.writerow(['JOBID=$SLURM_JOB_ID'])

        csv_writer.writerow(['echo -e "JobID: $JOBID\\n======"'])
        csv_writer.writerow(['echo "Time: `date`"'])
        csv_writer.writerow(['echo "Running on master node: `hostname`"'])
        csv_writer.writerow(['echo "Current directory: `pwd`"'])

        csv_writer.writerow(['if [ "$SLURM_JOB_NODELIST" ]; then'])
        csv_writer.writerow(['        export NODEFILE=`generate_pbs_nodefile`'])
        csv_writer.writerow(['        cat $NODEFILE | uniq > machine.file.$JOBID'])
        csv_writer.writerow(['        echo -e "\\nNodes allocated:\\n================"'])
        csv_writer.writerow([r"        echo `cat machine.file.$JOBID | sed -e 's/\..*$//g'`"])
        csv_writer.writerow(['fi'])

        csv_writer.writerow(['echo -e "\\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"'])

        csv_writer.writerow(['echo -e "\\nExecuting command:\\n==================\\n$CMD\\n"'])

        csv_writer.writerow(['eval $CMD'])

    subprocess.run(["dos2unix", f"{train_script_path}"])

def recursively_write_train_script(metadata_folder_path: pathlib.Path):
    for child_path in metadata_folder_path.glob("*"):
        if child_path.is_dir():
            recursively_write_train_script(child_path)
        else:
            if child_path.is_file():
                write_train_script(child_path)
            else:
                pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_folder_path', default = 'metadata_folder')
    args = parser.parse_args()

    metadata_file_path = pathlib.Path(args.metadata_folder_path)

    recursively_write_train_script(metadata_file_path)