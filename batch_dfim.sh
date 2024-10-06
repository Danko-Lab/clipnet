#!/bin/sh
#SBATCH --job-name=sp1_dfim
#SBATCH --ntasks=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --ntasks-per-core=1
#SBATCH --mem-per-gpu=64G
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --mail-type=all
#SBATCH --mail-user=ayh8@cornell.edu

# Script to run the training of the model on the PSC Bridges2 GPU Cluster

# Set up the environment

#mamba init
mamba activate clipnet
cd ~/storage/adamyhe/clipnet

python calculate_dfim.py \
    ../data/lcl/tfbs_sampling/sp1_tss_windows_reference_seq_oriented.fna.gz \
    ../data/lcl/tfbs_sampling/sp1_tss_windows_reference_seq_oriented_dfim_profile.npz \
    --gpu 0 --seed 47 --skip_check_additivity --mode profile