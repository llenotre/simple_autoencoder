import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

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

    def train(data):
        loss_fn = self.MSELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)

        for i, X in data.enumerate():
            Y = model(X)
            loss = self.loss_fn(X, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print("loss:", loss.item(), "; step:", i, "/", len(data))

model = Autoencoder().to(device)
print(model)

