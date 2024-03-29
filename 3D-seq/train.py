from unet3d import Unet3d
import torch
import torch.nn as nn
import numpy as np
from implicit import ImplicitEllipse,ImplicitUnion
from PIL import Image

SMOOTH = 1e-6

def iou(outputs: torch.Tensor, labels: torch.Tensor):
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return thresholded.mean()  # Or thresholded.mean() if you are interested in average across the batch


def random_ellipses(n, d, r_shift=0.3, r_fac=0.01):

    Es = list()

    for i in range(n):

        c = np.random.rand(d)
        a = 1.5*np.random.rand(d)
        # a = np.ones(d)
        r = r_shift + r_fac*np.random.rand(1)
        # r = 0.2

        Es.append(ImplicitEllipse(c, a, r))

    return ImplicitUnion(*Es)       


x = np.linspace(0, 1, 64)
y = np.linspace(0, 1, 64)
z = np.linspace(0, 1, 64)

grid = np.meshgrid(x, y, z)

n_ellipses_target = 3
n_ellipses_noise = 2
dim = 3

shape_target = random_ellipses(n_ellipses_target, dim)

value_target = shape_target(grid)
segmentation_target = shape_target.interior(grid, True)
image_target = -1*value_target*segmentation_target

shape_noise = random_ellipses(n_ellipses_noise, dim, 0.2, 0.1)
value_noise = shape_noise(grid)

segmentation_noise = shape_noise.interior(grid, True)
image_noise = -1*value_noise*segmentation_noise

image_blended = image_target + 0.3*image_noise

img = torch.from_numpy(image_blended).float()
mask = torch.from_numpy(segmentation_target).float()

unet = Unet3d()
optimizer = torch.optim.Adam(unet.parameters(),lr=0.0001)

n_img = 50

for i in range(n_img):
    shape_target = random_ellipses(n_ellipses_target, dim)

    value_target = shape_target(grid)
    segmentation_target = shape_target.interior(grid, True)
    image_target = -1*value_target*segmentation_target

    shape_noise = random_ellipses(n_ellipses_noise, dim, 0.2, 0.1)
    value_noise = shape_noise(grid)

    segmentation_noise = shape_noise.interior(grid, True)
    image_noise = -1*value_noise*segmentation_noise

    image_blended = image_target + 0.3*image_noise

    img = torch.from_numpy(image_blended).float()
    mask = torch.from_numpy(segmentation_target).float()[None,None,:,:,:]

    optimizer.zero_grad()

    out = unet(img[None,None,:,:,:])

    criterion = nn.BCEWithLogitsLoss()

    loss = criterion(out,mask)
    loss.backward()
    optimizer.step()
    print(loss)
    print("IOU: ",iou(out>0.5,mask>0))


