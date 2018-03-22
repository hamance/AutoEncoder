"""Data process & loader."""

import json
import os

import numpy as np
import skimage.io
import torch as t
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST #, FashionMNIST


class COCO(Dataset):
    def __init__(self, img_dir, input_json, transform=None, ):
        with open(input_json, 'r') as f:
            info = json.load(f)
        imgs = info['images']
        self.data = [os.path.join(img_dir, img) for img in imgs]
        self.transform = transform

    def __getitem__(self, index):
        img = self.data[index]
        I = skimage.io.imread(img)
        if len(I.shape) == 2:
            I = I[:,:,np.newaxis]
            I = np.concatenate((I,I,I), axis=2)

        I = I.astype('float32')/255.0
        I = t.from_numpy(I.transpose([2,0,1]))

        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.data)


def get_mnist_loader(opt):

    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = MNIST('./data', transform=img_transform, download=False)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

    return dataloader


# def get_fashion_mnist_loader(opt):

#     img_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])

#     dataset = FashionMNIST('./data', transform=img_transform, download=False)
#     dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

#     return dataloader


def get_coco_loader(opt):
    img_transform = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = COCO(opt.coco_dir, opt.input_json, transform=img_transform)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

    return dataloader




def get_loader(opt):
    if opt.dataset == 'mnist':
        return get_mnist_loader(opt)
    elif opt.dataset == 'fashionmnist':
        return get_fashion_mnist_loader(opt)
    elif opt.dataset == 'coco':
        return get_coco_loader(opt)
