from mpi4py import MPI
from Unet_dist import DistUnet2D

import distdl

def gen_dist_net():

    P_world = distdl.backend.backend.Partition(MPI.COMM_WORLD)
    P_world._comm.Barrier()

    return DistUnet2D(P_world)
