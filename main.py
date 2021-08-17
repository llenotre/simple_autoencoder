#!/usr/bin/python3

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

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

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

def layer_size(i: int) -> int:
    in_size = 1000
    encoder_size = 100

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
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.linear_relu_stack(x)

    def train(self, data):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)

        for i, (X, _) in enumerate(data):
            input_size = layer_size(0)
            X = transforms.Compose([
                transforms.Resize((input_size, input_size), transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ])(X)

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

        for i, (X, _) in enumerate(data):
            input_size = layer_size(0)
            X = transforms.Compose([
                transforms.Resize((input_size, input_size), transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ])(X)

            Y = self(X)
            total_loss += self.loss_fn(Y, X).item()

        print("Average loss:", total_loss / size)

if sys.argv[1] == '--train':
    model = Autoencoder().to(device)
    print(model)

    data = ImageDataset(path='data/test/')
    data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=32, shuffle=True, num_workers=8)

    epochs = 5
    for i in range(epochs):
        print("Epoch: ", i, "/", epochs)
        model.train(data)
        model.test(data)
    print("Done!")

    torch.save(model, 'model.pt')
else:
    model = torch.load('model.pt')
    print(model)

    image_path = sys.argv[1]
    image = Image.open(image_path)
    image_width, image_height = image.size
    input_size = layer_size(0)
    X = transforms.Compose([
        transforms.Resize((input_size, input_size), transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])(image)

    Y = model(X)
    Y = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_width, image_height), transforms.InterpolationMode.BILINEAR),
    ])(Y)

    fig = plt.figure(figsize=(8, 8))
    fig.add_subplot(1, 2, 1)
    plt.imshow(image)
    fig.add_subplot(1, 2, 2)
    plt.imshow(Y)
    plt.show()
