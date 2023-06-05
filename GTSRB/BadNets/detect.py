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
            model = PreActResNet18(num_classes=43)
        else:
            model = SENet18()
        model.load_state_dict(torch.load('save_model/'+args.model+'/model_portion0.1_'+str(i+1)+'.pth'))
        model = model.cuda()
        
        begin = time.time()
        noise = (torch.rand((1000,3,32,32))).float().cuda()
        with torch.no_grad():
            output = model(noise)
        _,predict = torch.max(output.data, dim=1)
        count = torch.zeros(43)
        for rank in predict:
            count[rank]+=1
        mean_count = count/count.mean()
        confidence = torch.softmax(output,dim=1)
        label = torch.zeros(43)
        count_softmax = []
        entropy = torch.zeros(43)
        for i in range(0,43):
            count_softmax.append(torch.zeros(43).cuda())
        for i in range(len(predict)):
            label[predict[i]]+=1
            count_softmax[predict[i]] += confidence[i]
        # print(label)
        mean_confidence = torch.zeros(43)
        zero_count = 0
        for i in range(0,43):
            if label[i] != 0:
                count_softmax[i] /= label[i]
                mean_confidence[i] = count_softmax[i][i]
                entropy[i]=sum(-count_softmax[i]*torch.log(count_softmax[i])).cpu().detach().item()
            else:
                zero_count+=1
        mean_confidence = entropy/sum(entropy)*(len(entropy)-zero_count)

        # mean_confidence = torch.where(torch.isnan(mean_confidence), torch.full_like(mean_confidence, 1), mean_confidence)
        mean_confidence[mean_confidence == 0.0] = 1.0
        # print(mean_count)
        # print(mean_confidence)

        support_detect = mean_count.ge(2).int()
        confidence_detect = mean_confidence.le(0.8).int()

        support_detect_label = support_detect.nonzero().view(-1).numpy().tolist()
        confidence_detect_label = confidence_detect.nonzero().view(-1).numpy().tolist()

        SAndC_label = list(set(support_detect_label).intersection(set(confidence_detect_label)))
        print('S Indicator Label Set: ',support_detect_label)
        print('C Indicator Label Set: ',confidence_detect_label)
        print('S&C Intersection: ',SAndC_label)

        if 0 in support_detect_label:
            Support_acc += 1

        if 0 in confidence_detect_label:
            Confidence_acc += 1

        if 0 in SAndC_label:
            SupportAndConfidence_acc += 1
        end = time.time()
        print(end-begin)
        sum_time += (end-begin)
        print('support:{}, confidence:{}, SupportAndConfidence_acc:{}'.format(Support_acc,Confidence_acc,SupportAndConfidence_acc))

print(sum_time/args.modelnum)