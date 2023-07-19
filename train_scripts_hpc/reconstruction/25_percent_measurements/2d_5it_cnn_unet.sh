#!/bin/bash
#SBATCH -J 2d_5it_cnn_unet
#SBATCH -A SCHONLIEB-SL3-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
