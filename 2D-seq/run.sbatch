#!/bin/bash

# 1 node, 1 mpi task, 4 cores (or openMP threads) available for that task
#SBATCH -N 1 --ntasks-per-node=1 -c 4
#SBATCH -t 02:30:00
#SBATCH -p normal_q

#SBATCH -A distdl

#Change to the directory from which the job was submitted
cd $SLURM_SUBMIT_DIR


#Load module
module load foss/2020a
module load SciPy-bundle/2020.03-foss-2020a-Python-3.8.2
module load PyTorch/1.4.0-foss-2020a-Python-3.8.2
module load mpi4py/3.0.2-gompi-2020a-timed-pingpong

mpirun -np 1 --map-by ppr:1:L3cache --bind-to l3cache -x OMP_NUM_THREADS=4 python test.py
