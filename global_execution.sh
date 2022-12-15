#!/bin/bash


# Load modules
module load intel impi

# Compile code
mpicc -Wall -o parallel_conv parallel_conv.c -lm

# Remove the previous results
rm images/output_* slurm_files/* output_files/*

# Single execution loop varying the number of processes
for ((i = 1; i <= 32; i++))
do
        sbatch -n $i --ntasks-per-node $i single_execution.sh $i
done