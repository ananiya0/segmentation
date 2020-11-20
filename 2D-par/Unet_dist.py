# Adapted from https://github.com/spctr01/UNet/blob/master/Unet.py
import numpy as np
import torch
import torch.nn as nn
from mpi4py import MPI
import distdl
from layers import DistributedNetworkOutput

def dual_conv(P_in, in_channels, out_channels):
    conv = nn.Sequential(
        distdl.nn.DistributedConv2d(P_in,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3,3),
                padding=(1,1)),
        distdl.nn.DistributedBatchNorm(P_in, num_features=out_channels),
        nn.ReLU(True),
        distdl.nn.DistributedConv2d(P_in,
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=(3,3),
                        padding=(1,1)),
        distdl.nn.DistributedBatchNorm(P_in, num_features=out_channels),
        nn.ReLU(True)
    )

    return conv

class DistUnet2D(distdl.nn.Module):

    def __init__(self, P_world):
        super(DistUnet2D, self).__init__()

        # Setup
        P_world._comm.Barrier()

        # Sticking with 4 workers/ranks for initial tests, change as needed
        P_base = P_world.create_partition_inclusive(np.arange(4))
        self.P_base = P_base

        # Partition used for input/output
        P_0 = P_base.create_partition_inclusive([0])
        P_root = P_0.create_cartesian_topology_partition([1, 1, 1, 1])

        # Layers needed as given in 2D seq example
        P_conv = P_base.create_cartesian_topology_partition([1, 1, 2, 2])

        # Maps input from one worker to the feature workers
        self.input_map = distdl.nn.DistributedTranspose(P_root, P_conv)

        # UNET LEFT SIDE
        self.dwn_conv1 = dual_conv(P_conv,3,64)
        self.dwn_conv2 = dual_conv(P_conv,64,128)
        self.dwn_conv3 = dual_conv(P_conv,128,256)
        self.dwn_conv4 = dual_conv(P_conv,256,512)
        self.dwn_conv5 = dual_conv(P_conv,512,1024)
        self.maxpool = distdl.nn.DistributedMaxPool2d(P_conv,
                                kernel_size=(2,2),
                            stride=(2,2))

    # UNET RIGHT SIDE
    # Ideally, these upsamps followed by post upsample convolutions
    # represented (psamp_conv) would be instead replaced with
    # distributed conv 2d's with even kernel_size and stride,
    # but distdl has an issue with that at the moment.
        self.upsample1 = distdl.nn.DistributedUpsample(P_conv, scale_factor=2)
        self.psamp_conv1 = distdl.nn.DistributedConv2d(P_conv,
                                                in_channels=1024,
                                                out_channels=512,
                                                kernel_size=(3,3),
                                                stride=(1,1))
        self.up_conv1 = dual_conv(P_conv, 1024, 512)

        self.upsample2 = distdl.nn.DistributedUpsample(P_conv, scale_factor=2)
        self.psamp_conv2 = distdl.nn.DistributedConv2d(P_conv,
                                                in_channels=512,
                                                out_channels=256,
                                                kernel_size=(3,3),
                                                stride=(1,1))
        self.up_conv2 = dual_conv(P_conv, 512, 256)

        self.upsample3 = distdl.nn.DistributedUpsample(P_conv, scale_factor=2)
        self.psamp_conv3 = distdl.nn.DistributedConv2d(P_conv,
                                                in_channels=256,
                                                out_channels=128,
                                                kernel_size=(3,3),
                                                stride=(1,1))
        self.up_conv3 = dual_conv(P_conv, 256, 128)

        self.upsample4 = distdl.nn.DistributedUpsample(P_conv, scale_factor=2)
        self.psamp_conv4 = distdl.nn.DistributedConv2d(P_conv,
                                                in_channels=128,
                                                out_channels=64,
                                                kernel_size=(3,3),
                                                stride=(1,1))
        self.up_conv4 = dual_conv(P_conv, 128, 64)

    #output

        self.out = distdl.nn.DistributedConv2d(P_conv,
                    in_channels=64,
                    out_channels=1,
                    kernel_size=(1,1))

        self.out = DistributedNetworkOutput(P_conv)

    def forward(self, img):

        x = self.input_map(img)

        x1 = self.dwn_conv1(x)
        x2 = self.maxpool(x1)
        x3 = self.dwn_conv2(x2)
        x4 = self.maxpool(x3)
        x5 = self.dwn_conv3(x4)
        x6 = self.maxpool(x5)
        x7 = self.dwn_conv4(x6)
        x8 = self.maxpool(x7)
        x9 = self.dwn_conv5(x8)

    # forward for left side

    #forward pass for Right side
        x = self.upsample1(x9)
        x = self.psamp_conv1(x)
        x = self.up_conv1(torch.cat([x,x7], 1))

        x = self.upsample2(x)
        x = self.psamp_conv2(x)
        x = self.up_conv2(torch.cat([x,x5], 1))

        x = self.upsample3(x)
        x = self.psamp_conv3(x)
        x = self.up_conv3(torch.cat([x,x3], 1))

        x = self.upsample4(x)
        x = self.psamp_conv4(x)
        x = self.up_conv4(torch.cat([x,x1], 1))

        x = self.out(x)

        return x


