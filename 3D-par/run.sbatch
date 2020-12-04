#!/bin/bash

# 1 node, 1 mpi task, 4 cores (or openMP threads) available for that task
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH -t 01:00:00
#SBATCH --partition=normal_q
#SBATCH --account=distdl

#Change to the directory from which the job was submitted
cd $SLURM_SUBMIT_DIR


#Load module
module load foss/2020a
module load SciPy-bundle/2020.03-gomkl-2020a-Python-3.8.2
module load PyTorch/1.6.0-gomkl-2020a-Python-3.8.2
module load mpi4py/3.0.2-gompi-2020a-timed-pingpong
module load matplotlib

mpirun -np 8 --map-by ppr:8:node --bind-to L3cache -x OMP_NUM_THREADS=16 python train.py
