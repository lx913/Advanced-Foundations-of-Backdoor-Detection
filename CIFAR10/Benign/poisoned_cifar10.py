import copy
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F


class PoisonedCIFAR10(Dataset):
    def __init__(self, dataset, trigger_label, portion=0.1, patch_size=5, atk='badnet', mode="train", dataname="cifar10"): 
        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        imgs = torch.empty([len(dataset), 3, 32, 32])
        labels = torch.empty([len(dataset)], dtype=torch.long)
        
        for idx in range(len(dataset)):
            imgs[idx] = transform(Image.fromarray(dataset.data[idx]))
            labels[idx] = torch.tensor(dataset.targets[idx])
        
        self.class_num = len(dataset.classes)
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx
        self.dataname = dataname
        self.mode = mode
        self.data, self.targets = self.add_trigger(imgs, labels, trigger_label, portion, patch_size, atk, mode, dataname)
        self.channels, self.width, self.height = self.__shape_info__()
        
    def __getitem__(self, item):
        img = self.data[item]
        label = self.targets[item]

        return img, label

    def __len__(self):
        return len(self.data)

    def __shape_info__(self):
        return self.data.shape[1:]
    

    def add_trigger(self, data, targets, trigger_label, portion, patch_size, atk, mode, dataname):
        print("## generate " + mode + " Bad Imgs")
        new_data = copy.deepcopy(data)
        new_targets = copy.deepcopy(targets)
        channels, width, height = new_data.shape[1:]
     
        perm = np.random.permutation(len(new_data))[0: int(len(new_data) * portion)]
        new_targets[perm] = trigger_label
        if atk == 'badnet':
            start_x = width - patch_size - 3
            start_y = height - patch_size - 3
            new_data[perm, :, start_x:start_x+patch_size, start_y:start_y+patch_size] = 1
        elif atk == 'blend':
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            ])
            pattern = transform(Image.open('blended_pattern/hello_kitty.jpeg').convert("RGB"))
            blended_rate = 0.1
            new_data[perm] = (1-blended_rate)*new_data[perm] + blended_rate*pattern   
        else:
            raise NotImplementedError()

        # for idx in perm: # if image in perm list, add trigger into img and change the label to trigger_label
        #     new_targets[idx] = trigger_label 
        #     start_x = width - patch_size - 3
        #     start_y = height - patch_size - 3
        #     new_data[idx, :, start_x:start_x+patch_size, start_y+patch_size] = 1

        print("Injecting Over: %d Bad Imgs, %d Clean Imgs (%.2f)" % (len(perm), len(new_data)-len(perm), portion))
       
        return new_data, new_targets

        
        