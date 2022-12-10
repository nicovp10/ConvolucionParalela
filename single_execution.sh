#!/bin/bash


#SBATCH -J parallel_convolution
#SBATCH --output=./report_files/%x-%j.out
#SBATCH -t 1:00:00
#SBATCH --mem=96G


# Determine the number of columns per block
C=8

# Execute the code
srun parallel_conv $C