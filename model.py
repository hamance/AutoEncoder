"""Model definition."""

import math

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

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = nn.Upsample(scale_factor=upsample)
        self.reflection_padding = int(np.floor(kernel_size / 2))
        if self.reflection_padding != 0:
            self.reflection_pad = nn.ReflectionPad2d(self.reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

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
        

    def forward(self, x):
        # import ipdb; ipdb.set_trace()
        x = self.encoder(x)
        x = self.decoder(x)
        return x



class Linear(nn.Module):
    ''' Simple Linear layer with xavier init '''
    def __init__(self, d_in, d_out, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(d_in, d_out, bias=bias)
        init.xavier_normal(self.linear.weight)

    def forward(self, x):
        return self.linear(x)

class Bottle(nn.Module):
    ''' Perform the reshape routine before and after an operation '''

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0]*size[1], -1))
        return out.view(size[0], size[1], -1)

class BottleLinear(Bottle, Linear):
    ''' Perform the reshape routine before and after a linear projection '''
    pass


class EncodeBlock(nn.Module):

    def __init__(self, d_in, d_hid, d_out, s_k, stride):
        super(EncodeBlock, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(d_in, d_hid, s_k, stride=stride, padding=math.floor(s_k/2)),
            nn.BatchNorm2d(d_hid),
            nn.ReLU(True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(d_hid, d_out, s_k, stride=stride, padding=math.floor(s_k/2)),
            nn.BatchNorm2d(d_hid),
            nn.ReLU(True)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(d_in, d_out, 1, 1, 0),
            nn.AvgPool2d(s_k, stride*2, padding=math.floor(s_k/2))
        )

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        xd = self.block3(x1)
        return x2 + xd


class DecodeBlock(nn.Module):

    def __init__(self, d_in, d_out, s_k, stride, upsample):
        super(DecodeBlock, self).__init__()
        self.block = nn.Sequential(
            UpsampleConvLayer(d_in, d_out, s_k, stride, math.floor(s_k/2), upsample),
            nn.BatchNorm2d(d_out),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.block(x)


class autoencoder2(nn.Module):
    def __init__(self):
        super(autoencoder2, self).__init__()
        self.encoder = nn.Sequential(                  # b, 3, 512, 512
            EncodeBlock(3, 16, 32, 5, 2),       # b, 32, 128, 128
            EncodeBlock(32, 16, 3, 5, 4),      # b, 3, 8, 8
        )
        self.decoder = nn.Sequential(
            DecodeBlock(3, 16, 5, 2, 8),    # b, 16, 32, 32
            DecodeBlock(16, 16, 5, 2, 8),   # b, 16, 128, 128
            DecodeBlock(16, 3, 3, 1, 4)     # b, 3, 512, 512
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
