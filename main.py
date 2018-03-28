"""Main.py"""

import os
import time

import torch as t
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image

from config import opt
from data import get_loader
from model import autoencoder, autoencoder2


def to_img(x):
    x = 0.5 * (x+1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 512, 512)
    return x

def to_img2(x):
    x = 0.5 * (x+1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

def main(**kwargs):
    opt.parse(kwargs)
    opt.show()
    
    if opt.dataset == 'mnist':
        model = autoencoder2().cuda()
    else:
        model = autoencoder().cuda()
    criterion = nn.MSELoss()
    optimizer = t.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=1e-5)
    dataloader = get_loader(opt)

    if opt.load_from is not None:
        state = t.load(opt.load_from)
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])

    for epoch in range(opt.num_epochs):
        for ii, data in enumerate(dataloader):
            img, _ = data
            img = Variable(img).cuda()

            output = model(img)
            loss = criterion(output, img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ii % 100 == 0:
                print("step [{}:{}], loss {:.4f}".format(epoch+1, ii, loss.data[0]))

        print("epoch [{}/{}], loss{:.4f}".format(epoch+1, opt.num_epochs, loss.data[0]))

        if epoch % 10 == 0:
            pic = to_img(output.cpu().data)
            save_image(pic, os.path.join(opt.save_dir, 'dc_img', 'img_{}.png'.format(epoch)))
            
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            t.save(state, os.path.join(opt.save_dir, 'model', 'ckpt_{}.pt'.format(epoch)))


if __name__ == '__main__':
    import fire
    fire.Fire(main)
