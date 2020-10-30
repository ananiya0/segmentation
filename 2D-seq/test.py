from PIL.Image import NEAREST
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from cvcloader import CVC
import torch
from network import Unet

transform = transforms.Compose([
    transforms.Resize([256,256]),
    transforms.ToTensor(),
])

target_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize([256,256],interpolation=NEAREST),
    transforms.ToTensor(),
])

Unet = Unet()
Unet.load_state_dict(torch.load("./UNET.pt"))

test_set = CVC("./CVC/test/Original","./CVC/test/Ground Truth",
    transform=transform,target_transform=target_transform)

test = DataLoader(test_set,batch_size=1,shuffle=True)

for data in test:
    x,y = data
    output = Unet(x)
    save_image(x,"img.png")
    save_image(output,"pred.png")
    save_image(y,"mask.png")
    break