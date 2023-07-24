#!/bin/bash
#SBATCH -J full_dataset_2d_5it_sinogram_MSE_0_loss
#SBATCH -A SCHONLIEB-SL3-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mail-type=NONE
#SBATCH --no-requeue
#SBATCH -p ampere
numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')
. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
. /etc/profile.d/modules.sh
module load miniconda/3
source /home/ev373/.bashrc
conda activate all_in_one
application="/home/ev373/.conda/envs/all_in_one/bin/python"
options="experiences.py --platform hpc --metadata_path metadata_folder/reconstruction/25_percent_measurements/full_dataset_2d_5it_sinogram_MSE_0_loss.json"
workdir="/home/ev373/work/all_in_one"
export OMP_NUM_THREADS=1
np=$[${numnodes}*${mpi_tasks_per_node}]
CMD="$application $options"
cd $workdir
echo -e "Changed directory to `pwd`.\n"
JOBID=$SLURM_JOB_ID
echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"
if [ "$SLURM_JOB_NODELIST" ]; then
        export NODEFILE=`generate_pbs_nodefile`
        cat $NODEFILE | uniq > machine.file.$JOBID
        echo -e "\nNodes allocated:\n================"
        echo `cat machine.file.$JOBID | sed -e 's/\..*$//g'`
fi
echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"
echo -e "\nExecuting command:\n==================\n$CMD\n"
eval $CMD
