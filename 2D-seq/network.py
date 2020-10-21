import torch
import torchvision
from torchvision import transforms, datasets
from data_loader import get_data_loaders
import torch.nn as nn
import torch.nn.functional as fnn
import torch.optim as optim

train, test = get_data_loaders(10,download=True,dummy=False)

""" class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1*28*28,64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,64)
        self.fc4 = nn.Linear(64,10)

    def forward(self,x):
        x = fnn.relu(self.fc1(x))
        x = fnn.relu(self.fc2(x))
        x = fnn.relu(self.fc3(x))
        x = self.fc4(x)
        return fnn.log_softmax(x,dim=1)

net = Net()

optimizer = optim.Adam(net.parameters(),lr=0.001)

EPOCHS = 3

for epoch in range (EPOCHS):
    for data in train:
        x,y = data
        net.zero_grad()
        output = net(x.view(-1,28*28))
        loss = fnn.nll_loss(output,y)
        loss.backward()
        optimizer.step()

correct = 0
total = 0

with torch.no_grad():
    for data in test:
        x,y = data
        output = net(x.view(-1,28*28))
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1
    
print("Accuracy: ", correct/total) """