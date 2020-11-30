from implicit import ImplicitEllipse, ImplicitUnion
from unet3d import Unet3d
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def iou(outputs: torch.Tensor, labels: torch.Tensor):
    SMOOTH = 1e-6
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


def best_slice(volume):
    max = 0
    for i in range(np.size(volume,0)):
        area = np.mean(volume[max,:,:])
        new_area = np.mean(volume[i,:,:])
        max = i * (new_area > area) + max * (area >= new_area)
    return max


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

n_img = 5

unet = Unet3d()
unet.load_state_dict(torch.load("./UNET3d.pt"))

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

    out = unet(img[None,None,:,:,:])

    sig = nn.Sigmoid()
    out = sig(out)
    out = out > 0.5
    mask = mask > 0

    print("IOU: ",iou(out,mask))

    out = np.squeeze(out.detach().numpy())
    mask = np.squeeze(mask.detach().numpy())

    i = best_slice(mask)
    print(i)
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(mask[i,:,:])
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.imshow(out[i,:,:])
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("plot2.png")