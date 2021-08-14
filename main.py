import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, ToPILImage
import torch.nn.functional as F

import matplotlib.pyplot as plt

import glob
from PIL import Image

import os
from os import listdir

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

class ImageDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.path + '/' + os.listdir(self.path)[index]
        x = ToTensor()(Image.open(image_path))
        F.interpolate(x, 10000)
        if self.transform is not None:
            x = self.transform(x)
        return x, x

    def __len__(self):
        return len(self.path)

def layer_size(i: int) -> int:
    in_size = 10000
    encoder_size = 1000

    encoder_dist = (10 - i)
    if encoder_dist < 0:
        encoder_dist = -encoder_dist
    slope = (in_size - encoder_size) / 20

    return int(encoder_size + encoder_dist * slope)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.flatten = nn.Flatten()
        layers = []
        for i in range(0, 20):
            layers.append(nn.Linear(layer_size(i), layer_size(i + 1)))
            layers.append(nn.ReLU())
        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def train(self, data):
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)

        for i, (X, _) in enumerate(data):
            Y = self(X)
            loss = self.loss_fn(Y, X)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print("loss:", loss.item(), "; step:", i, "/", len(data))

model = Autoencoder().to(device)
print(model)

data = ImageDataset(path='data/crowdsource_images-00000-of-00010/')
data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=32, shuffle=True, num_workers=4)

#i = 0
#fig = plt.figure(figsize=(8, 8))
#for x, _ in data:
#    fig.add_subplot(5, 5, i + 1)
#    plt.imshow(x)
#
#    i += 1
#    if i >= 25:
#        break
#plt.show()

model.train(data)
