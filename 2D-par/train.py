import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL.Image import NEAREST
from cvcloader import CVC
from network import Unet



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

train_data = CVC("../CVC_data/train/Original", "../CVC_data/train/Ground Truth", 
    transform=transform,target_transform=target_transform)

train = DataLoader(train_data,batch_size=1,shuffle=True)

Unet = Unet()
optimizer = torch.optim.Adam(Unet.parameters(),lr=0.0001)
EPOCHS = 15
for epoch in range(EPOCHS):
    for data in train:
        x,y = data
        optimizer.zero_grad()
        output = Unet(x)
        criterion = torch.nn.BCEWithLogitsLoss()
        #y = y[None,:,:,:]
        loss = criterion(output,y)
        #print(dice_loss(output,y))
        loss.backward()
        optimizer.step()
        print(loss)
    save_image(x,"img.png")
    save_image(output,"pred.png")
    save_image(y,"mask.png")
        
torch.save(Unet.state_dict(),"./UNET.pt") 
