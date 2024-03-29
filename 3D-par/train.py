import torch
import torch.nn as nn
import numpy as np
from implicit import ImplicitEllipse,ImplicitUnion
from gen_dist_net import gen_dist_net
#import matplotlib.pyplot as plt
from mpi4py import MPI
import distdl

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


def gen_data(grid, n_ellipses, n_noise, dim):
    n_ellipses_target = n_ellipses
    n_ellipses_noise = n_noise
    dim = dim

    shape_target = random_ellipses(n_ellipses_target, dim)

    value_target = shape_target(grid)
    segmentation_target = shape_target.interior(grid, True)
    image_target = -1*value_target*segmentation_target

    shape_noise = random_ellipses(n_ellipses_noise, dim, 0.2, 0.1)
    value_noise = shape_noise(grid)

    segmentation_noise = shape_noise.interior(grid, True)
    image_noise = -1*value_noise*segmentation_noise

    image_blended = image_target + 0.3*image_noise

    img = torch.from_numpy(image_blended).float()[None,None,:,:,:]
    mask = torch.from_numpy(segmentation_target).float()[None,None,:,:,:]

    return img,mask


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

unet = gen_dist_net()
P_base = unet.P_base

MPI.COMM_WORLD.Barrier()

parameters = [p for p in unet.parameters()]

if not parameters:
    parameters = [torch.nn.Parameter(torch.zeros(1))]

optimizer = torch.optim.Adam(parameters,lr=0.0001)
criterion = nn.BCEWithLogitsLoss()

n_img = 50
batch_size = 1
iter = 0

start = MPI.Wtime()

for i in range(n_img):
    if P_base.rank == 0:
        iter += 1
        img,mask = gen_data(grid,3,2,3)

        for batch in range(batch_size-1):
            img1,mask1 = gen_data(grid,3,2,3)

            img = torch.cat([img,img1],dim=0)
            mask = torch.cat([mask,mask1],dim=0)
    else:
        img = distdl.utilities.torch.zero_volume_tensor(batch_size)
        mask = distdl.utilities.torch.zero_volume_tensor(batch_size)

    optimizer.zero_grad()

    out = unet(img)

    if P_base.rank == 0:
        loss = criterion(out,mask)
        print(loss)
        print("IOU: ",iou(out>0.5,mask>0))
    else:
        loss = out.clone()
    
    loss.backward()
    optimizer.step()

    """ if P_base.rank == 0:
        if iter%10 == 0:
            sig = nn.Sigmoid()
            out = sig(out)
            out = out > 0.5
            mask = mask > 0

            out = np.squeeze(out.detach()[0].numpy())
            mask = np.squeeze(mask.detach()[0].numpy())

            i = best_slice(mask)
            plt.figure()

            plt.subplot(1,2,1)
            plt.imshow(mask[i,:,:])
            plt.colorbar(fraction=0.046, pad=0.04)

            plt.subplot(1,2,2)
            plt.imshow(out[i,:,:])
            plt.colorbar(fraction=0.046, pad=0.04)

            plt.tight_layout()
            plt.savefig("plot.png") """

if P_base.rank == 0:
    print(MPI.Wtime() - start, " seconds elapsed")