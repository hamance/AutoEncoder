"""Model definition."""

import torch as t
import torch.nn as nn

from torch.autograd import Variable


class NoiseLayer(nn.Module):

    def __init__(self, is_training=True, mu=0, std=0.05):
        super(NoiseLayer, self).__init__()
        self.mu = mu
        self.std = std
        self.is_training = is_training

    def forward(self, x):
        if self.is_training:
            noise = Variable(x.data.new(x.size()).normal_(self.mu, self.std))
            return x + noise
        return x


class autoencoder(nn.Module):

    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(                  # b, 1, 512, 512
            nn.Conv2d(3, 16, 5, stride=4, padding=2),  # b, 16, 128, 128
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            NoiseLayer(self.training),
            nn.Conv2d(16, 64, 5, stride=4, padding=2),  # b, 16, 32, 32
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            NoiseLayer(self.training),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # b, 16, 16, 16
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            NoiseLayer(self.training),
            nn.Conv2d(128, 64, 3, stride=2, padding=1),  # b, 16, 16, 16
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            NoiseLayer(self.training),
            nn.Conv2d(64, 32, 3, stride=2, padding=1),  # b, 16, 8, 8
            # nn.BatchNorm2d(8),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 64, 3, stride=2, padding=1, output_padding=1),  # b, 8, 16, 16
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 128, 3, stride=2, padding=1, output_padding=1),  # b, 16, 32, 32
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # b, 16, 32, 32
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 16, 5, stride=4, padding=2, output_padding=3),  # b, 16, 128, 128
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 5, stride=4, padding=2, output_padding=3),  # b, 16, 512, 512
            # nn.BatchNorm2d(3),
            nn.Tanh()
        )

    def forward(self, x):
        # import ipdb; ipdb.set_trace()
        x = self.encoder(x)
        x = self.decoder(x)
        return x



class autoencoder2(nn.Module):
    def __init__(self):
        super(autoencoder2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x