#!/bin/bash


# Load modules
module load intel impi

# Compile code
mpicc -Wall -o parallel_conv parallel_conv.c -lm

rm images/o* report_files/*

# Single execution loop varying the number of processes
for ((i = 1; i <= 32; i = i+1)); 
do
        for ((j = 8; j <= 8; j = j+1));
        do
                sbatch -n $i --ntasks-per-node $i single_execution.sh $j
        done
done