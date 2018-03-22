"""Data process & loader."""

from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader



def get_loader(opt):

    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = MNIST('./data', transform=img_transform, download=True)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

    return dataloader
