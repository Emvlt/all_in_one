#!/bin/bash
#SBATCH -J lpd_unet_joint
#SBATCH -A SCHONLIEB-SL3-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
