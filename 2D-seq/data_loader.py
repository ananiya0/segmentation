# Adapted from https://github.com/activatedgeek/LeNet-5/blob/master/run.py
# ---
from PIL.Image import NEAREST
import numpy as np
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation



#custom collate_fn
def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [data,target]

class SquarePad:
    def __call__(self,image):
        w, h = image.size
        wp = int((512 - w))
        hp = int((512 - h))
        padding = (0, 0, wp, hp)
        return F.pad(image, padding, 0, 'constant')

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

    data_train = VOCSegmentation('./data', year = "2012", image_set="train",
                       download=download,
                       transform=transforms.Compose([
                           transforms.Resize([256,256]),
                           #SquarePad(),
                           transforms.ToTensor()
                           ]),
                           target_transform=transforms.Compose([
                           transforms.Resize([256,256],interpolation=NEAREST),
                           #SquarePad(),
                           transforms.ToTensor()
                           ]))

    data_test = VOCSegmentation('./data', year = "2012", image_set="val",
                      download=download,
                      transform=transforms.Compose([
                          transforms.Resize([256,256]),
                          #SquarePad(),
                          transforms.ToTensor()
                          ]),
                          target_transform=transforms.Compose([
                          transforms.Resize([256,256],interpolation=NEAREST),
                          #SquarePad(),
                          transforms.ToTensor()
                          ]))

    if not dummy:
        train_loader = DataLoader(data_train,
                                  batch_size=batch_size,
                                  #collate_fn=my_collate,
                                  shuffle=True,
                                  num_workers=0)
        test_loader = DataLoader(data_test,
                                 batch_size=batch_size,
                                 #collate_fn=my_collate,
                                 num_workers=0)
    else:
        train_loader = DummyLoader(batch_size, 60000)
        test_loader = DummyLoader(batch_size, 10000)

    return train_loader, test_loader

# ---
