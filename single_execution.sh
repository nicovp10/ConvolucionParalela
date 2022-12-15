#!/bin/bash


#SBATCH -J parallel_convolution
#SBATCH --output=./slurm_files/%x-%j.out
#SBATCH -t 10:00
#SBATCH --mem=32G


# Different number of columns per block
# 1 2 4 8 16 32 64 
C_values=(256)

# Number of executions
ITER=1


# Loop through the different values of C
for C in ${C_values[@]}
do
    output=output_files/output_$1.txt
    echo -e "\nC: $C" >> $output

    # Execute the code ITER times
    for ((i = 1; i <= $ITER; i++))
    do
        echo -e "\n\niteration: $i\n" >> $output
        srun parallel_conv $C >> $output
    done
done