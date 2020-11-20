# Adapted from https://github.com/activatedgeek/LeNet-5/blob/master/run.py
# ---
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from cvc_load import CVC
from PIL.Image import NEAREST

import distdl


class DummyLoader:

    def __init__(self, batch_size, n_data_points):

        self.batch_size = batch_size
        self.n_data_points = n_data_points

        self.n_loaded = 0

        self.mod_batch_size = n_data_points % batch_size
        self.n_batches = n_data_points // batch_size

    def __iter__(self):
        for i in range(self.n_batches):
            yield distdl.utilities.torch.zero_volume_tensor(self.batch_size), torch.zeros(self.batch_size)
        yield distdl.utilities.torch.zero_volume_tensor(self.mod_batch_size), torch.zeros(self.mod_batch_size)


def get_data_loaders(batch_size, download=False, dummy=False):
    transform = transforms.Compose([
        transforms.Resize([256,256]),
        transforms.ToTensor(),
    ])

    target_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize([256,256],interpolation=NEAREST),
        transforms.ToTensor(),
    ])

    data_train = CVC('../CVC_data/train/Original',
                       '../CVC_data/train/Ground Truth',
                       transform=transform, target_transform=target_transform)

    data_test = CVC('../CVC_data/test/Original',
                      '../CVC_data/test/Ground Truth',
                      transform=transform, target_transform=target_transform)

    if not dummy:
        train_loader = DataLoader(data_train,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=1)
        test_loader = DataLoader(data_test,
                                 batch_size=batch_size,
                                 num_workers=1)
    else:
        train_loader = DummyLoader(batch_size, 491)
        test_loader = DummyLoader(batch_size, 123)

    return train_loader, test_loader

# ---
