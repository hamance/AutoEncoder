"""Model definition."""

import numpy as np
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



class UpsampleConvLayer(nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = nn.Upsample(scale_factor=upsample)
        self.reflection_padding = int(np.floor(kernel_size / 2))
        if self.reflection_padding != 0:
            self.reflection_pad = nn.ReflectionPad2d(self.reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        if self.upsample:
            x = self.upsample_layer(x)
        if self.reflection_padding != 0:
            x = self.reflection_pad(x)
        out = self.conv2d(x)
        return out


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
        self.decoder2 = nn.Sequential(
            UpsampleConvLayer(32, 64, 3, stride=1, upsample=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            UpsampleConvLayer(64, 128, 3, stride=1, upsample=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            UpsampleConvLayer(128, 64, 3, stride=1, upsample=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            UpsampleConvLayer(64, 16, 3, stride=1, upsample=4),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            UpsampleConvLayer(16, 3, 3, stride=1, upsample=4),
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
