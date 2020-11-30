# Adapted from https://github.com/spctr01/UNet/blob/master/Unet.py
import torch
import torch.nn as nn
import torch.nn.functional as fnn
import numpy as np
from mpi4py import MPI
import distdl
from output import DistributedNetworkOutput

#double 3x3 convolution 
def dual_conv(P_in,in_channel, out_channel):
    conv = nn.Sequential(
        distdl.nn.DistributedConv3d(P_in,in_channel, out_channel, kernel_size=3,padding=1),
        distdl.nn.DistributedBatchNorm3d(P_in,out_channel),
        nn.ReLU(inplace=True),
        distdl.nn.DistributedConv3d(P_in,out_channel, out_channel, kernel_size=3,padding=1),
        distdl.nn.DistributedBatchNorm3d(P_in,out_channel),
        nn.ReLU(inplace=True)
    )
    return conv

def upconv(P_in,in_channel,out_channel,kernel_size):
    up = nn.Sequential(
        distdl.nn.DistributedUpsample(P_in,scale_factor=2),
        distdl.nn.DistributedConv3d(P_in,in_channel, out_channel, kernel_size=kernel_size)
    )
    return up


class DistUnet3D(nn.Module):
    def __init__(self,P_World):
        super(DistUnet3D, self).__init__()

        # Setup
        P_World._comm.Barrier()
        P_base = P_World.create_partition_inclusive(np.arange(4))
        self.P_base = P_base

        # Partition used for input/output
        P_0 = P_base.create_partition_inclusive([0])
        P_root = P_0.create_cartesian_topology_partition([1, 1, 1, 1, 1])

        # Partition for Conv layers
        P_conv = P_base.create_cartesian_topology_partition([1, 1, 2, 2, 2])

        # Maps input from one worker to the feature workers
        self.input_map = distdl.nn.DistributedTranspose(P_root, P_conv)

        # Left side (contracting path)
        self.dwn_conv1 = dual_conv(P_conv, 1, 64)
        self.dwn_conv2 = dual_conv(P_conv, 64, 128)
        self.dwn_conv3 = dual_conv(P_conv, 128, 256)
        self.dwn_conv4 = dual_conv(P_conv, 256, 512)
        self.dwn_conv5 = dual_conv(P_conv, 512, 1024)
        self.maxpool = distdl.nn.DistributedMaxPool3d(P_conv,kernel_size=2, stride=2)

        #Right side  (expansion path) 
        #transpose convolution is used shown as green arrow in architecture image
        self.trans1 = upconv(P_conv, 1024,512, kernel_size=1)
        self.up_conv1 = dual_conv(P_conv, 1024,512)
        self.trans2 = upconv(P_conv, 512,256, kernel_size=1)
        self.up_conv2 = dual_conv(P_conv, 512,256)
        self.trans3 = upconv(P_conv, 256, 128, kernel_size=1)
        self.up_conv3 = dual_conv(P_conv, 256,128)
        self.trans4 = upconv(P_conv, 128,64, kernel_size=1)
        self.up_conv4 = dual_conv(P_conv, 128,64)

        #output layer
        self.out_conv = distdl.nn.DistributedConv3d(P_conv, 64, 1, kernel_size=1)
        self.output_map = distdl.nn.DistributedTranspose(P_conv,P_root)
        self.out = DistributedNetworkOutput(P_conv)

    def forward(self, image):

        x = self.input_map(image)

        #forward pass for Left side
        x1 = self.dwn_conv1(x)
        x2 = self.maxpool(x1)
        x3 = self.dwn_conv2(x2)
        x4 = self.maxpool(x3)
        x5 = self.dwn_conv3(x4)
        x6 = self.maxpool(x5)
        x7 = self.dwn_conv4(x6)
        x8 = self.maxpool(x7)
        x9 = self.dwn_conv5(x8)
        

        #forward pass for Right side\
        x = self.trans1(x9)
        x = self.up_conv1(torch.cat([x,x7], 1))
        
        x = self.trans2(x)
        x = self.up_conv2(torch.cat([x,x5], 1))

        x = self.trans3(x)
        x = self.up_conv3(torch.cat([x,x3], 1))

        x = self.trans4(x)
        x = self.up_conv4(torch.cat([x,x1], 1))
        
        x = self.out_conv(x)
        x = self.output_map(x)
        x = self.out(x)
        
        return x


