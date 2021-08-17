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
    in_size = 900
    encoder_size = 100
    slope = (in_size - encoder_size) / 10

    encoder_dist = abs(10 - i)
    return int(encoder_size + encoder_dist * slope)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        encode_layers = []
        for i in range(0, 10):
            encode_layers.append(nn.Linear(layer_size(i), layer_size(i + 1)))
            encode_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encode_layers)

        decode_layers = []
        for i in range(10, 20):
            decode_layers.append(nn.Linear(layer_size(i), layer_size(i + 1)))
            decode_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*decode_layers)

        self.loss_fn = nn.MSELoss()

    def get_size(self):
        return layer_size(0)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)

    def train(self, data):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        for i, (X, _) in enumerate(data):
            X = torch.flatten(X)
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
            X = torch.flatten(X)
            Y = self(X)
            total_loss += self.loss_fn(Y, X).item()

        print("Average loss:", total_loss / size)

if sys.argv[1] == '--train':
    model = Autoencoder().to(device)
    print(model)

    input_image_size = int(math.sqrt(model.get_size()))
    data = ImageDataset(path='data/test/', transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((input_image_size, input_image_size), transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ]))
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
    input_image_size = int(math.sqrt(model.get_size()))
    X = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((input_image_size, input_image_size), transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])(image)
    X = torch.flatten(X)

    Y = model.encode(X)
    Y = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_height, image_width), transforms.InterpolationMode.BILINEAR),
    ])(Y)

    fig = plt.figure(figsize=(8, 8))
    fig.add_subplot(1, 2, 1)
    plt.imshow(image)
    fig.add_subplot(1, 2, 2)
    plt.imshow(Y)
    plt.show()
