#!/bin/bash


#SBATCH -J parallel_convolution
#SBATCH --output=./report_files/%x-%j.out
#SBATCH -t 1:00:00
#SBATCH --mem=4G


# Execute the code
srun parallel_conv