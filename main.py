#!/usr/bin/python3

import math
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor, ToPILImage
import torch.nn.functional as F

import matplotlib.pyplot as plt

import glob
from PIL import Image

import os
from os import listdir

#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
print("Using {} device".format(device))



dim = 90



class ImageDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.path + '/' + os.listdir(self.path)[index]
        x = Image.open(image_path)
        if self.transform is not None:
            x = self.transform(x)
        return x, x

    def __len__(self):
        return len(os.listdir(self.path))

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encode_conv0 = nn.Conv2d(3, 6, kernel_size=3)
        self.encode_relu0 = nn.ReLU()
        self.encode_pool0 = nn.MaxPool2d(2, return_indices=True)

        self.encode_conv1 = nn.Conv2d(6, 16, kernel_size=3)
        self.encode_relu1 = nn.ReLU()
        self.encode_pool1 = nn.MaxPool2d(2, return_indices=True)

        self.decode_unpool0 = nn.MaxUnpool2d(2)
        self.decode_conv0 = nn.ConvTranspose2d(16, 6, kernel_size=3)
        self.decode_relu0 = nn.ReLU()

        self.decode_unpool1 = nn.MaxUnpool2d(2)
        self.decode_conv1 = nn.ConvTranspose2d(6, 3, kernel_size=3)
        self.decode_relu1 = nn.ReLU()

        self.loss_fn = nn.MSELoss()

    def encode(self, x):
        x = self.encode_conv0(x)
        x = self.encode_relu0(x)
        x, indices0 = self.encode_pool0(x)

        x = self.encode_conv1(x)
        x = self.encode_relu1(x)
        x, indices1 = self.encode_pool1(x)
        return x, indices0, indices1

    def decode(self, x, indices0, indices1):
        x = self.decode_unpool0(x, indices1)
        x = self.decode_conv0(x)
        x = self.decode_relu0(x)

        x = self.decode_unpool1(x, indices0)
        x = self.decode_conv1(x)
        x = self.decode_relu1(x)
        return x

    def forward(self, x):
        x, indices0, indices1 = self.encode(x)
        return self.decode(x, indices0, indices1)

    def do_train(self, data):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        self.train()
        for i, (X, _) in enumerate(data):
            X = X.view(1, 3, dim, dim).to(device)
            Y = self(X)
            loss = self.loss_fn(Y, X)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print("loss:", loss.item(), "; step:", i, "/", len(data))

    def test(self, data):
        size = len(data)
        total_loss = 0.

        self.eval()
        with torch.no_grad():
            for i, (X, _) in enumerate(data):
                X = X.view(1, 3, dim, dim).to(device)
                Y = self(X)
                total_loss += self.loss_fn(Y, X).item()

        print("Average loss:", total_loss / size)



if sys.argv[1] == '--train':
    model = Autoencoder().to(device)
    print(model)

    data = ImageDataset(path='data/test/', transform = transforms.Compose([
        transforms.Resize((dim, dim), transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ]))
    data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=32, shuffle=True, num_workers=4)

    epochs = 10
    for i in range(epochs):
        print("Epoch: ", i, "/", epochs)
        model.do_train(data)
        model.test(data)
    print("Done!")

    torch.save(model.state_dict(), 'model.pth')
else:
    model = Autoencoder()
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    print(model)

    image_path = sys.argv[1]
    image = Image.open(image_path)
    image_width, image_height = image.size
    X = transforms.Compose([
        transforms.Resize((dim, dim), transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])(image)
    shape = X.shape

    with torch.no_grad():
        X = X.view(1, 3, dim, dim).to(device)
        Y = model(X)
    Y = Y.view(shape)
    Y = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((dim, dim), transforms.InterpolationMode.BILINEAR),
    ])(Y)

    fig = plt.figure(figsize=(8, 8))
    fig.add_subplot(1, 2, 1)
    plt.imshow(image)
    fig.add_subplot(1, 2, 2)
    plt.imshow(Y)
    plt.show()
