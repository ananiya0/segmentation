# Adapted from https://github.com/activatedgeek/LeNet-5/blob/master/run.py
# ---
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST

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
            yield distdl.utilities.torch.zero_volume_tensor(self.batch_size), np.zeros(self.batch_size)
        yield distdl.utilities.torch.zero_volume_tensor(self.mod_batch_size), np.zeros(self.mod_batch_size)


def get_data_loaders(batch_size, download=False, dummy=False):

    data_train = MNIST('./data',
                       download=download,
                       transform=transforms.Compose([transforms.ToTensor()]))
    data_test = MNIST('./data',
                      train=False,
                      download=download,
                      transform=transforms.Compose([transforms.ToTensor()]))

    if not dummy:
        train_loader = DataLoader(data_train,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=0)
        test_loader = DataLoader(data_test,
                                 batch_size=batch_size,
                                 num_workers=0)
    else:
        train_loader = DummyLoader(batch_size, 60000)
        test_loader = DummyLoader(batch_size, 10000)

    return train_loader, test_loader

# ---
