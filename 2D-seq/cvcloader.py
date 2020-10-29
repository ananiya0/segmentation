import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms,datasets
from torchvision.utils import save_image

class CVC(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_transform = target_transform
        self.imgs = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.img_dir, self.imgs[idx])
        mask_loc = os.path.join(self.mask_dir, self.masks[idx])
        image = Image.open(img_loc)
        mask = Image.open(mask_loc)
        tensor_image = self.transform(image)[None,:,:,:]
        tensor_mask = self.target_transform(mask)[None:,:,:]
        return tensor_image, tensor_mask

