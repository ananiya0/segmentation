# Adapted from https://github.com/spctr01/UNet/blob/master/Unet.py
from PIL.Image import NEAREST
import torch
import torchvision
from torchvision import transforms, datasets
from torchvision.transforms.transforms import Grayscale, Resize
from data_loader import get_data_loaders
from torchvision.utils import save_image
import torch.nn as nn
import torch.nn.functional as fnn
import torch.optim as optim
from cvcloader import CVC

def dice_loss(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))


transform = transforms.Compose([
    transforms.Resize([256,256]),
    transforms.ToTensor(),
])

target_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize([256,256],interpolation=NEAREST),
    transforms.ToTensor(),
])

train = CVC("./CVC/train/Original", "./CVC/train/Ground Truth", 
    transform=transform,target_transform=target_transform)

#double 3x3 convolution 
def dual_conv(in_channel, out_channel):
    conv = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3,padding=1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace= True),
        nn.Conv2d(out_channel, out_channel, kernel_size=3,padding=1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace= True),
    )
    return conv


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        # Left side (contracting path)
        self.dwn_conv1 = dual_conv(3, 64)
        self.dwn_conv2 = dual_conv(64, 128)
        self.dwn_conv3 = dual_conv(128, 256)
        self.dwn_conv4 = dual_conv(256, 512)
        self.dwn_conv5 = dual_conv(512, 1024)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        #Right side  (expansion path) 
        #transpose convolution is used showna as green arrow in architecture image
        self.trans1 = nn.ConvTranspose2d(1024,512, kernel_size=2, stride=2)
        self.up_conv1 = dual_conv(1024,512)
        self.trans2 = nn.ConvTranspose2d(512,256, kernel_size=2, stride=2)
        self.up_conv2 = dual_conv(512,256)
        self.trans3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = dual_conv(256,128)
        self.trans4 = nn.ConvTranspose2d(128,64, kernel_size=2, stride=2)
        self.up_conv4 = dual_conv(128,64)

        #output layer
        self.out = nn.Conv2d(64, 1, kernel_size=1)
        #self.last = nn.Softmax(dim=1)

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
        

        #forward pass for Right side
        x = self.trans1(x9)
        x = self.up_conv1(torch.cat([x,x7], 1))
        
        x = self.trans2(x)
        x = self.up_conv2(torch.cat([x,x5], 1))

        x = self.trans3(x)
        x = self.up_conv3(torch.cat([x,x3], 1))

        x = self.trans4(x)
        x = self.up_conv4(torch.cat([x,x1], 1))
        
        x = self.out(x)
        #x = self.last(x)
        
        return x


Unet = Unet()
optimizer = optim.Adam(Unet.parameters(),lr=0.0001)
EPOCHS = 1
for epoch in range(EPOCHS):
    for data in train:
        x,y = data
        optimizer.zero_grad()
        output = Unet(x)
        criterion = nn.BCEWithLogitsLoss()
        y = y[None,:,:,:]
        loss = criterion(output,y)
        #print(dice_loss(output,y))
        loss.backward()
        optimizer.step()
        print(loss)
        save_image(x,"img.png")
        save_image(output,"pred.png")
        save_image(y,"mask.png")
        
        
        
        
        

