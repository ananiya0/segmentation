#!/bin/bash

# 1 node, 4 mpi tasks, 4 cores (or openMP threads) available for that task
#SBATCH -N 1 --ntasks-per-node=8 -c 8
#SBATCH -t 04:00:00
#SBATCH -p normal_q

#SBATCH -A distdl

#Change to the directory from which the job was submitted
cd $SLURM_SUBMIT_DIR


#Load module
module load foss/2020a
module load SciPy-bundle/2020.03-gomkl-2020a-Python-3.8.2
module load PyTorch/1.6.0-gomkl-2020a-Python-3.8.2
module load mpi4py/3.0.2-gompi-2020a-timed-pingpong

mpirun -np 8 -x OMP_NUM_THREADS=1 python train.py
