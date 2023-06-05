import torchvision
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import torch
import torchvision
import argparse

from model_template.preactres import PreActResNet18
from model_template.senet import SENet18
import pandas as pd
import time

parser = argparse.ArgumentParser(description='Detect".')
parser.add_argument('-savepath', type=str, default='save_model/', help='how many models you want to train')
parser.add_argument('-model', type=str, default='resnet', help='determine the model architecture')
parser.add_argument('-modelnum', type=int, default=30, help='how many models you want to train')

args = parser.parse_args()

if __name__ == '__main__':
    Support_acc = 0
    Confidence_acc = 0
    SupportAndConfidence_acc = 0
    sum_time = 0

    for i in range(args.modelnum):
        if args.model == 'resnet':
            model = PreActResNet18(num_classes=10)
        else:
            model = SENet18()
        model.load_state_dict(torch.load('save_model/'+args.model+'_fine-tune_0.05/clean_model_'+str(i+1)+'.pth'))
        model = model.cuda()
        
        begin = time.time()
        noise = (torch.rand((1000,3,32,32))).float().cuda()
        with torch.no_grad():
            output = model(noise)
        _,predict = torch.max(output.data, dim=1)
        count = torch.zeros(10)
        for rank in predict:
            count[rank]+=1
        confidence = abs(output.mean(dim=0)).cpu()
        mean_confidence = confidence/confidence.mean()
        mean_count = count/count.mean()
        count_bar =  max(mean_count)*1.05
        confidence_bar = max(mean_confidence)*1.05

        # print(mean_count,count_bar)
        # print(mean_confidence,confidence_bar)

        if args.model == 'resnet':
            model = PreActResNet18(num_classes=10)
        else:
            model = SENet18()
        model.load_state_dict(torch.load('save_model/'+args.model+'/clean_model_'+str(i+1)+'.pth'))
        model = model.cuda()

        noise = (torch.rand((1000,3,32,32))).float().cuda()
        with torch.no_grad():
            output = model(noise)
        _,predict = torch.max(output.data, dim=1)
        count = torch.zeros(10)
        for rank in predict:
            count[rank]+=1
        confidence = abs(output.mean(dim=0)).cpu()
        mean_confidence = confidence/confidence.mean()
        mean_count = count/count.mean()
        # print(mean_count,count_bar)
        # print(mean_confidence,confidence_bar)

        support_detect = mean_count.ge(count_bar).int()
        confidence_detect = mean_confidence.ge(confidence_bar).int()

        support_detect_label = support_detect.nonzero().view(-1).numpy().tolist()
        confidence_detect_label = confidence_detect.nonzero().view(-1).numpy().tolist()

        SAndC_label = list(set(support_detect_label).intersection(set(confidence_detect_label)))
        print('S Indicator Label Set: ',support_detect_label)
        print('C Indicator Label Set: ',confidence_detect_label)
        print('S&C Intersection: ',SAndC_label)
        
        if len(support_detect_label) == 0:
            Support_acc += 1

        if len(confidence_detect_label) == 0:
            Confidence_acc += 1

        if len(SAndC_label) == 0:
            SupportAndConfidence_acc += 1

        end = time.time()
        print(end-begin)
        sum_time += (end-begin)
        print('support:{}, confidence:{}, SupportAndConfidence_acc:{}'.format(Support_acc,Confidence_acc,SupportAndConfidence_acc))

print(sum_time/args.modelnum)