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
    def __init__(self, in_size: int, out_size: int):
        super(Autoencoder, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.encoder_depth = 5

        encode_layers = []
        for i in range(0, self.encoder_depth):
            encode_layers.append(nn.Linear(self.layer_size(i), self.layer_size(i + 1)))
            encode_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encode_layers)

        decode_layers = []
        for i in range(self.encoder_depth, self.encoder_depth * 2):
            decode_layers.append(nn.Linear(self.layer_size(i), self.layer_size(i + 1)))
            decode_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*decode_layers)

        self.loss_fn = nn.MSELoss()

    def layer_size(self, i: int) -> int:
        slope = (self.in_size - self.out_size) / self.encoder_depth
        encoder_dist = abs(self.encoder_depth - i)
        return int(self.out_size + encoder_dist * slope)

    def get_input_size(self):
        return self.in_size

    def get_output_size(self):
        return self.out_size

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def do_train(self, data):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        self.train()
        for i, (X, _) in enumerate(data):
            X = X.to(device)
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

        self.eval()
        with torch.no_grad():
            for i, (X, _) in enumerate(data):
                X = X.to(device)
                X = torch.flatten(X)
                Y = self(X)
                total_loss += self.loss_fn(Y, X).item()

        print("Average loss:", total_loss / size)



channels_count = 3
in_dim = 30
out_dim = 10

if sys.argv[1] == '--train':
    model = Autoencoder(in_dim * in_dim * channels_count, out_dim * out_dim * channels_count).to(device)
    print(model)

    data = ImageDataset(path='data/test/', transform = transforms.Compose([
        transforms.Resize((in_dim, in_dim), transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ]))
    data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=32, shuffle=True, num_workers=4)

    epochs = 5
    for i in range(epochs):
        print("Epoch: ", i, "/", epochs)
        model.do_train(data)
        model.test(data)
    print("Done!")

    torch.save(model.state_dict(), 'model.pth')
else:
    model = Autoencoder(in_dim * in_dim * channels_count, out_dim * out_dim * channels_count)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    print(model)

    image_path = sys.argv[1]
    image = Image.open(image_path)
    image_width, image_height = image.size
    X = transforms.Compose([
        transforms.Resize((in_dim, in_dim), transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])(image)
    shape = X.shape
    X = torch.flatten(X)

    with torch.no_grad():
        X = X.to(device)
        Y = model.encoder(X)
    Y = Y.view((channels_count, out_dim, out_dim))
    Y = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_height, image_width), transforms.InterpolationMode.BILINEAR),
    ])(Y)

    with torch.no_grad():
        X = X.to(device)
        decoded = model(X)
    decoded = decoded.view(shape)
    decoded = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_height, image_width), transforms.InterpolationMode.BILINEAR),
    ])(decoded)

    fig = plt.figure(figsize=(8, 8))
    fig.add_subplot(1, 3, 1)
    plt.imshow(image)
    fig.add_subplot(1, 3, 2)
    plt.imshow(Y)
    fig.add_subplot(1, 3, 3)
    plt.imshow(decoded)
    plt.show()
