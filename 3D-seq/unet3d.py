# Adapted from https://github.com/spctr01/UNet/blob/master/Unet.py
import torch
import torch.nn as nn
import torch.nn.functional as fnn


#double 3x3 convolution 
def dual_conv(in_channel, out_channel):
    conv = nn.Sequential(
        nn.Conv3d(in_channel, out_channel, kernel_size=3,padding=1),
        nn.BatchNorm3d(out_channel),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channel, out_channel, kernel_size=3,padding=1),
        nn.BatchNorm3d(out_channel),
        nn.ReLU(inplace=True)
    )
    return conv

def upconv(in_channel,out_channel,kernel_size):
    up = nn.Sequential(
        nn.Upsample(scale_factor=2),
        nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size)
    )
    return up


class Unet3d(nn.Module):
    def __init__(self):
        super(Unet3d, self).__init__()

        # Left side (contracting path)
        self.dwn_conv1 = dual_conv(1, 64)
        self.dwn_conv2 = dual_conv(64, 128)
        self.dwn_conv3 = dual_conv(128, 256)
        self.dwn_conv4 = dual_conv(256, 512)
        self.dwn_conv5 = dual_conv(512, 1024)
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        #Right side  (expansion path) 
        #transpose convolution is used shown as green arrow in architecture image
        self.trans1 = upconv(1024,512, kernel_size=1)
        self.up_conv1 = dual_conv(1024,512)
        self.trans2 = upconv(512,256, kernel_size=1)
        self.up_conv2 = dual_conv(512,256)
        self.trans3 = upconv(256, 128, kernel_size=1)
        self.up_conv3 = dual_conv(256,128)
        self.trans4 = upconv(128,64, kernel_size=1)
        self.up_conv4 = dual_conv(128,64)

        #output layer
        self.out = nn.Conv3d(64, 1, kernel_size=1)

    def forward(self, image):

        #forward pass for Left side
        x1 = self.dwn_conv1(image)
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
        
        x = self.out(x)
        
        return x


