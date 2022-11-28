#!/bin/bash


#SBATCH -J P3
#SBATCH -t 1:00:00
#SBATCH --mem=4G
#SBATCH -n 2
#SBATCH --ntasks-per-node 2


module load intel impi

srun p3