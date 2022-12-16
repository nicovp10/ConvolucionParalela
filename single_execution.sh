#!/bin/bash


#SBATCH -J parallel_convolution
#SBATCH --output=./slurm_files/%x-%j.out
#SBATCH -t 10:00
#SBATCH --mem=32G


# Different number of columns per block
C_values=(1 15 55 105)

# Number of executions
ITER=5


# Loop through the different values of C
for C in ${C_values[@]}
do
    output=output_files/output_$1.txt
    echo -e "C: $C\n" >> $output

    # Execute the code ITER times
    for ((i = 1; i <= $ITER; i++))
    do
        echo -e "iteration: $i\n" >> $output
        srun parallel_conv $C >> $output
    done
done