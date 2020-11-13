# Adapted from https://github.com/spctr01/UNet/blob/master/Unet.py
import numpy as np
import torch
import torch.nn as nn
from mpi4py import MPI
import distdl


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
	# I believe we will need to define a dist batch norm operation but I am not
	# sure how this would come into place here.
    P_fc_in = P_base_lo.create_cartesian_topology_partition([1, 2])
    P_fc_out = P_base_hi.create_cartesian_topology_partition([1, 2])
    P_fc_mtx = P_base.create_cartesian_topology_partition([2, 2])

	
    """
    Layers needed for seq are (in order):
	conv2d
	batchnorm2d
	relu
	conv2d
	batchnorm2d
	relu
    """

    net = torch.nn.Sequential(distdl.nn.DistributedConv2d(P_conv, # Do I need a distributed transpose first?
							  							in_channels=3,
							  							out_channels=64,
							  							kernel_size=(3,3),
							  							padding=(1,1)),
			    	# IDK HOW TO BATCH NORM, we need to replace that placeholder with
			    	# an object that i don't really understand check out the docs
			    	# We could also just do a distributed batchnorm i think
			    	distdl.nn.DistributedUpsample(P_conv,
								buffer_manager=None,
							    size=None,
							    scale_factor=None,
							    mode='linear',
							    align_corners=False),
			      	torch.nn.ReLU(),
			      	distdl.nn.DistributedConv2d(P_conv,
                                                in_channels=64,
                                                out_channels=64,
                                                kernel_size=(3,3),
                                                padding=(1,1)),
			      	# NEEDS ANOTHER BATCH NORM
			      	distdl.nn.DistributedUpsample(P_conv,
							    buffer_manager=None,
							    size=None,
							    scale_factor=None,
							    mode='linear',
							    align_corners=False),
			      	torch.nn.ReLU())
