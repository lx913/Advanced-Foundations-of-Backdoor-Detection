from torch.utils.data import DataLoader
import torch
import torchvision
from torchvision.transforms import transforms
import numpy as np
import pandas as pd
from torch import nn
from torch.autograd import Variable
from torchvision.transforms import ToPILImage
import os.path
import os
import argparse

from model_template.preactres import PreActResNet18
from model_template.senet import SENet18
from poisoned_gtsrb import PoisonedGTSRB


parser = argparse.ArgumentParser(description='Clean & BackDoor Models Training".')
parser.add_argument('-dataset', default='GTSRB', help='Which dataset to use (mnist or cifar10, default: mnist)')
parser.add_argument('-optimizer', default='sgd', help='Which optimizer to use (sgd or adam, default: sgd)')
parser.add_argument('-trigger_label', type=int, default=0, help='The NO. of trigger label (int, range from 0 to 10, default: 0)')
parser.add_argument('-epoch', type=int, default=100, help='Number of epochs to train backdoor model, default: 50')
parser.add_argument('-lr', type=float, default=0.01, help='learning rate, default: 0.01')
parser.add_argument('-batchsize', type=int, default=128, help='Batch size to split dataset, default: 64')
parser.add_argument('-download', action='store_false', help='Do you want to download data ( default false, if you add this param, then download)')
parser.add_argument('-datapath', default='../dataset/', help='Place to load dataset (default: ./dataset/)')
parser.add_argument('-poisoned_portion', type=float, default=0.1, help='posioning portion (float, range from 0 to 1, default: 0.1)')
parser.add_argument('-patch_size', type=int, default=5, help='triger size')
parser.add_argument('-atk', type=str, default='badnet', help='which attack you want to implement')
parser.add_argument('-modelnum', type=int, default=30, help='how many models you want to train')
parser.add_argument('-savepath', type=str, default='save_model/', help='how many models you want to train')
parser.add_argument('-model', type=str, default='resnet', help='determine the model architecture')

args = parser.parse_args()

print(args.__dict__)

def load_init_data(dataname, download, dataset_path):
    # download dataset to dir /dataset/dataname
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    if not os.path.exists(dataset_path+dataname):
        os.mkdir(dataset_path+dataname)
    train_data = torchvision.datasets.GTSRB(root=dataset_path+dataname,split="train",download=download)
    test_data = torchvision.datasets.GTSRB(root=dataset_path+dataname,split="test",download=download)
    return train_data, test_data


def create_backdoor_data_loader(dataname, train_data, test_data, trigger_label, posioned_portion, atk, batch_size):
    if dataname == 'GTSRB':
        train_data_clean = PoisonedGTSRB(train_data, trigger_label, portion=0, patch_size = args.patch_size, atk=args.atk, mode="train", dataname=dataname)
        train_data_poison = PoisonedGTSRB(train_data, trigger_label, portion=posioned_portion, patch_size = args.patch_size, atk=args.atk, mode="train", dataname=dataname)
        test_data_clean = PoisonedGTSRB(test_data, trigger_label, portion=0, mode="test", patch_size = args.patch_size, atk=args.atk, dataname=dataname)
        test_data_poison = PoisonedGTSRB(test_data, trigger_label, portion=1, mode="test", patch_size = args.patch_size, atk=args.atk, dataname=dataname)

        train_data_clean_loader = DataLoader(dataset=train_data_clean, batch_size=batch_size, num_workers=8, shuffle=True)
        train_data_poison_loader = DataLoader(dataset=train_data_poison, batch_size=batch_size, num_workers=8, shuffle=True)
        test_data_clean_loader = DataLoader(dataset=test_data_clean, batch_size=batch_size, num_workers=8, shuffle=True)
        test_data_poison_loader = DataLoader(dataset=test_data_poison, batch_size=batch_size, num_workers=8, shuffle=True)  # shuffle 随机化

    return train_data_clean_loader, train_data_poison_loader, test_data_clean_loader, test_data_poison_loader

def train(model, data_loader, criterion, optimizer):
    model.train()
    running_loss = 0
    num_examples = 0
    num_correct = 0
    for step, (batch_x, batch_y) in enumerate(data_loader):
        if torch.cuda.is_available():
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
        # print(batch_x)
        
        optimizer.zero_grad()
        output = model(batch_x) 
        loss = criterion(output, batch_y)
        _, predicted = torch.max(output.data, 1)
        num_examples += batch_y.size(0)
        num_correct += (predicted == batch_y).sum().item()

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss, num_correct/num_examples*100.0

def eval(model, data_loader):
    model.eval()
    num_examples = 0
    num_correct = 0
    for step, (batch_x, batch_y) in enumerate(data_loader):
        if torch.cuda.is_available():
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
        
        output = model(batch_x)
        _, predicted = torch.max(output.data, 1)
        num_examples += batch_y.size(0)
        num_correct += (predicted == batch_y).sum().item()

    return num_correct/num_examples*100.0


if __name__ == "__main__":
    print("# --------------------------read dataset: %s --------------------------" % args.dataset)
    train_data, test_data = load_init_data(dataname=args.dataset, download=args.download, dataset_path=args.datapath)
    print("# ------------------------normal dataset: %s loaded ! --------------------------" % args.dataset)

    print("# --------------------------construct poisoned dataset--------------------------")
    print("# --------------------------%s attack!--------------------------" % args.atk)
    
    train_data_clean_loader, train_data_poison_loader, test_data_clean_loader, test_data_poison_loader = create_backdoor_data_loader(args.dataset, train_data,test_data,args.trigger_label,args.poisoned_portion,args.atk,args.batchsize)


    print("# --------------------------begin training model--------------------------")
    if args.dataset == 'GTSRB':
        num_classes = 43
    else:
        num_classes = 10
        
    if not os.path.exists(args.savepath):
        os.mkdir(args.savepath)
    save_path = args.savepath+args.model
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        

    train_loader = train_data_poison_loader
    
    
    
    print("# --------------------------"+args.atk+" models training")
    best_performance = []
    for num in range(args.modelnum):
        print("# --------------------------%s-th model training --------------------------" % str(num+1))
        if args.model == 'resnet':
            model = PreActResNet18(num_classes=num_classes)
        else:
            model = SENet18()
        if torch.cuda.is_available():
            model = model.cuda()

        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=4e-4)

        criterion = torch.nn.CrossEntropyLoss()

        best_attack_acc = 0
        best_clean_acc = 0
        for i in range(args.epoch):
            loss, acc_train = train(model, train_loader, criterion, optimizer)
            acc_test_clean = eval(model, test_data_clean_loader)
            acc_test_poison = eval(model, test_data_poison_loader)
            print("# EPOCH%d   loss: %.4f  training acc: %.4f, clean testing acc: %.4f, poison testing acc: %.4f\n" \
          % (i, loss, acc_train, acc_test_clean, acc_test_poison))

            if acc_test_poison >= best_attack_acc and acc_test_clean >= 90:
                best_attack_acc = acc_test_poison
                best_clean_acc = acc_test_clean
                print("beset BA & ASR! Model Saved!")
                torch.save(model.state_dict(),save_path+'/model_portion'+str(args.poisoned_portion)+'_'+str(num+1)+'.pth')

        best_performance.append([best_clean_acc,best_attack_acc])

    res = pd.DataFrame(columns=['BA','ASR'],data=best_performance)
    res.to_csv(save_path+'/result.csv',index=True,encoding='utf-8')

                
