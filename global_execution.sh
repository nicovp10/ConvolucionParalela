#!/bin/bash


# Load modules
module load intel impi

# Compile code
mpicc -o parallel_conv parallel_conv.c -lm

# Single execution loop varying the number of processes
for ((i = 2; i <= 32; i = i+1)); 
do
        sbatch -n $i --ntasks-per-node $i single_execution.sh
done