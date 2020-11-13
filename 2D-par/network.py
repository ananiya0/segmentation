# Adapted from https://github.com/spctr01/UNet/blob/master/Unet.py
import numpy as np
import torch
import torch.nn as nn
from mpi4py import MPI
import distdl
from layers import dist_dual_conv


def gen_dist_net():

    # Setup
    P_world = distdl.backend.backend.Partition(MPI.COMM_WORLD)
    P_world.comm.Barrier()

    # Sticking with 4 workers/ranks for initial tests, change as needed
    P_base = P_world.create_partition_inclusive(np.arange(4))

    # I/o partitions
    P_0 = P_base.create_partition_inclusive([0])
    P_root = P_0.create_cartesian_topology_partition([1, 1, 1, 1])
    P_root_2d = P_0.create_cartesian_topology_partition([1, 1])

    # Disjoint partitions of the base used for fully connected layer input/output
    P_base_lo = P_base.create_partition_inclusive(np.arange(0, 2))
    P_base_hi = P_base.create_partition_inclusive(np.arange(2, 4))        

    # Layers needed as given in 2D seq example
    P_conv = P_base.create_cartesian_topology_partition([1, 1, 2, 2])

    # I think we won't need these because we aren't dealing with flatten aat all
    #P_fc_in = P_base_lo.create_cartesian_topology_partition([1, 2])
    #P_fc_out = P_base_hi.create_cartesian_topology_partition([1, 2])
    #P_fc_mtx = P_base.create_cartesian_topology_partition([2, 2])

	
    """
    Layers needed for seq are (in order):
	conv2d
	batchnorm2d
	relu
	conv2d
	batchnorm2d
	relu
    """

    # Beginning of the left-hand side
    net = torch.nn.Sequential(dist_dual_conv(P_conv, 3, 64),
        dist_dual_conv(P_conv, 64, 128),
	dist_dual_conv(P_conv, 128, 256),
	dist_dual_conv(P_conv, 256, 512),
	dist_dual_conv(P_conv, 512, 1024), 
	distdl.nn.DistributedMaxPool2d(P_conv,
            kernel_size=(2, 2),
            stride=(2, 2)), # Beginning of the right-hand side
        distdl.nn.DistributedUpsample(P_conv),
	dist_dual_conv(P_conv, 1024, 512),
        distdl.nn.DistributedUpsample(P_conv),
        dist_dual_conv(P_conv, 512, 256),
	distdl.nn.DistributedUpsample(P_conv),
        dist_dual_conv(P_conv, 256, 128),
	distdl.nn.DistributedUpsample(P_conv),
	dist_dual_conv(P_conv, 128, 64),
	
	# OUTPUT LAYER
	distdl.DistributedConv2d(64,1, kernel_size=(1,1))
    )
    return P_base, net
