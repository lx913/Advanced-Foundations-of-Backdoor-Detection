import os
import numpy as np
from glob import glob
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import bchlib


class StegaData(Dataset):
    def __init__(self, dataset, trigger_label, size=(32, 32)):
        BCH_POLYNOMIAL = 137
        BCH_BITS = 5
        bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

        data = bytearray(str(trigger_label) + ' '*(7-len(str(trigger_label))), 'utf-8')
        ecc = bch.encode(data)
        packet = data + ecc

        packet_binary = ''.join(format(x, '08b') for x in packet)
        secret = [int(x) for x in packet_binary]
        secret.extend([0, 0, 0, 0])
        
        transform = transforms.Compose([
            transforms.Resize((size)),
            transforms.ToTensor()
        ])

        imgs = torch.empty([len(dataset), 3, 32, 32])
        
        
        for idx, (path, _) in enumerate(dataset._samples):
            imgs[idx] = transform(Image.open(path).convert("RGB"))
        
        self.secret = secret
        self.data = imgs
        self.secret_size = len(secret)

    def __getitem__(self, idx):
        img_cover = self.data[idx]
        
        secret = torch.tensor(self.secret).float()

        return secret, img_cover

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    # dataset = StegaData(data_path='F:\\VOCdevkit\\VOC2012\\JPEGImages')
    # print(len(dataset))
    # img_cover, secret = dataset[10]
    # print(type(img_cover), type(secret))
    # print(img_cover.shape, secret.shape)

    dataset = StegaData(data_path=r'E:\dataset\mirflickr', secret_size=100, size=(400, 400))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, pin_memory=True)
    image_input, secret_input = next(iter(dataloader))
    print(type(image_input), type(secret_input))
    print(image_input.shape, secret_input.shape)
    print(image_input.max())
