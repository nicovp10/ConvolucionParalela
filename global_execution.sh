#!/bin/bash


# Load modules
module load intel impi

# Compile code
mpicc -o p3 P3.c -lm

# Single execution loop varying the number of processes
for ((i = 2; i <= 2; i = i+1)); 
do
        sbatch -n $i --ntasks-per-node $i single_execution.sh
done