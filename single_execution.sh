#!/bin/bash


#SBATCH -J parallel_convolution
#SBATCH --output=./report_files/%x-%j.out
#SBATCH -t 1:00:00
#SBATCH --mem=96G


# Number of executions
ITER=5

# Number of columns per block
C=8


# Execute the code ITER times
for ((i = 1; i <= $ITER; i++))
do
    echo -e "\niteration: $i\n"
    srun parallel_conv $C
done