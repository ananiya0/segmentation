from mpi4py import MPI
from Unet3D import DistUnet3D

import distdl

def gen_dist_net():

    P_world = distdl.backend.backend.Partition(MPI.COMM_WORLD)
    P_world._comm.Barrier()

    return DistUnet3D(P_world)
