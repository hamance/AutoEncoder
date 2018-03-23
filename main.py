"""Main.py"""

import torch as t
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image

from config import opt
from data import get_loader
from model import autoencoder

import time

def to_img(x):
    x = 0.5 * (x+1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 512, 512)
    return x


def main(**kwargs):
    opt.parse(kwargs)
    opt.show()
    
    model = autoencoder().cuda()
    criterion = nn.MSELoss()
    optimizer = t.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=1e-5)
    dataloader = get_loader(opt)

    for epoch in range(opt.num_epochs):
        for data in dataloader:
            img, _ = data
            img = Variable(img).cuda()

            output = model(img)
            loss = criterion(output, img)

            loss.backward()
            optimizer.step()
            et = time.time()

        print("epoch [{}/{}], loss{:.4f}".format(epoch+1, opt.num_epochs, loss.data[0]))

        if epoch % 10 == 0:
            pic = to_img(output.cpu().data)
            save_image(pic, './dc_img/img_{}.png'.format(epoch))


if __name__ == '__main__':
    import fire
    fire.Fire(main)
