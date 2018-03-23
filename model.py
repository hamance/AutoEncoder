"""Model definition."""

import torch as t
import torch.nn as nn

from torch.autograd import Variable


class autoencoder(nn.Module):

    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(                  # b, 1, 512, 512
            nn.Conv2d(3, 16, 5, stride=4, padding=2),  # b, 16, 128, 128
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 16, 5, stride=4, padding=2),  # b, 16, 32, 32
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 16, 3, stride=2, padding=1),  # b, 16, 16, 16
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 16, 8, 8
            # nn.BatchNorm2d(16),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 8, 3, stride=2, padding=1, output_padding=1),  # b, 8, 16, 16
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 16, 3, stride=2, padding=1, output_padding=1),  # b, 16, 32, 32
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 16, 5, stride=4, padding=2, output_padding=3),  # b, 16, 128, 128
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 5, stride=4, padding=2, output_padding=3),  # b, 16, 512, 512
            # nn.BatchNorm2d(16),
            nn.Tanh()
        )

    def forward(self, x):
        # import ipdb; ipdb.set_trace()
        x = self.encoder(x)
        x = self.decoder(x)
        return x
