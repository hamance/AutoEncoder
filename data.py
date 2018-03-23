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
    def __init__(self, img_dir, input_json, mode='train', transform=None, train_only=False):
        with open(input_json, 'r') as f:
            info = json.load(f)
        self.split_ix = {'train':[], 'val':[], 'test':[]}
        for ix in range(len(info['images'])):
            img = info['images'][ix]
            if img['split'] == 'train':
                self.split_ix['train'].append(ix)
            elif img['split'] == 'val':
                self.split_ix['val'].append(ix)
            elif img['split'] == 'test':
                self.split_ix['test'].append(ix)
            elif train_only == 0: # restval
                self.split_ix['train'].append(ix)

        imgs = self.split_ix[mode]
        self.data = [os.path.join(img_dir, info['images'][img]['file_path']) for img in imgs]
        self.transform = transform

    def __getitem__(self, index):
        # img = self.data[index]
        # I = skimage.io.imread(img)
        # if len(I.shape) == 2:
        #     I = I[:,:,np.newaxis]
        #     I = np.concatenate((I,I,I), axis=2)

        # I = I.astype('float32')/255.0
        # # I = t.from_numpy(I.transpose([2,0,1]))
        # I = I.transpose([2, 0, 1])
        # import ipdb; ipdb.set_trace()
        # if self.transform is not None:
        #     img = self.transform(I)
        # return img
        img = self.data[index]
        I = Image.open(img).convert('RGB')
        # import ipdb; ipdb.set_trace()
        # if len(I.size) == 2:
        #     I = np.asarray(I)
        #     I = I[:,:,np.newaxis]
        #     I = np.concatenate((I, I, I), axis=2)
        if self.transform is not None:
            I = self.transform(I)
        return I, t.FloatTensor([0])


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
        # transforms.Scale(224),
        transforms.RandomSizedCrop(512),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = COCO(opt.coco_dir, opt.input_json, transform=img_transform)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=16)

    return dataloader




def get_loader(opt):
    if opt.dataset == 'mnist':
        return get_mnist_loader(opt)
    # elif opt.dataset == 'fashionmnist':
    #     return get_fashion_mnist_loader(opt)
    elif opt.dataset == 'coco':
        return get_coco_loader(opt)



if __name__ == '__main__':
    import fire
    from config import opt

    opt.coco_dir = 'g:\\image_caption\\zips\\coco\\2014'
    opt.input_json = 'g:\\image_caption\\coco\\lrt\\cocotalk.json'
    opt.batch_size = 1
    
    coco_loader = get_coco_loader(opt)
    
    for tt, (i, j) in enumerate(coco_loader):
        if (i.size() != (1, 3, 512, 512)):
            print(i.size())
        if tt%100 == 0:
            print(tt)
    print("Done.")
