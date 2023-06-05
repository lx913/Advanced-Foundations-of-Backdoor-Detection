import json
import os
import shutil
from time import time

import config
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from model_template.preactres import PreActResNet18
from model_template.senet import SENet18
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import RandomErasing
from utils.dataloader import PostTensorTransform, get_dataloader
from utils.utils import progress_bar
import pandas as pd


def get_model(opt):
    netC = None
    optimizerC = None
    schedulerC = None

    if opt.model == "resnet":
        netC = PreActResNet18(num_classes=opt.num_classes).to(opt.device)
    else:
        netC = SENet18().to(opt.device)

    # Optimizer
    optimizerC = torch.optim.SGD(netC.parameters(), opt.lr_C, momentum=0.9, weight_decay=5e-4)

    # Scheduler
    schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, opt.schedulerC_milestones, opt.schedulerC_lambda)

    return netC, optimizerC, schedulerC


def train(netC, optimizerC, schedulerC, train_dl, epoch, opt, num):
    print(" Train:")
    netC.train()
    rate_bd = opt.pc
    total_loss_ce = 0
    total_sample = 0
    total_correct = 0
    criterion_CE = torch.nn.CrossEntropyLoss()

    transforms = PostTensorTransform(opt).to(opt.device)
    total_time = 0

    avg_acc_cross = 0

    for batch_idx, (inputs, targets) in enumerate(train_dl):
        optimizerC.zero_grad()

        inputs, total_targets = inputs.to(opt.device), targets.to(opt.device)
        
        total_inputs = transforms(inputs)
        
        start = time()
        total_preds = netC(total_inputs)
        total_time += time() - start

        loss_ce = criterion_CE(total_preds, total_targets)

        loss = loss_ce
        loss.backward()

        optimizerC.step()

        total_sample += inputs.shape[0]
        total_loss_ce += loss_ce.detach()

        total_correct += torch.sum(torch.argmax(total_preds, dim=1) == total_targets)
        
        avg_acc = total_correct * 100.0 / total_sample

        avg_loss_ce = total_loss_ce / total_sample


        progress_bar(
            batch_idx,
            len(train_dl),
            "CE Loss: {:.4f} | Acc: {:.4f}".format(avg_loss_ce, avg_acc),
        )
        
        if batch_idx >= opt.ft_ratio*len(train_dl):
            print(batch_idx,"/",len(train_dl)," over ratio: ",opt.ft_ratio)
            break

    schedulerC.step()


def eval(
    netC,
    optimizerC,
    schedulerC,
    test_dl,
    noise_grid,
    identity_grid,
    best_clean_acc,
    best_bd_acc,
    best_cross_acc,
    epoch,
    opt,
    num
):
    print(" Eval:")
    netC.eval()

    total_sample = 0
    total_clean_correct = 0
    total_bd_correct = 0
    total_cross_correct = 0
    total_ae_loss = 0

    criterion_BCE = torch.nn.BCELoss()

    for batch_idx, (inputs, targets) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            bs = inputs.shape[0]
            total_sample += bs

            # Evaluate Clean
            preds_clean = netC(inputs)
            total_clean_correct += torch.sum(torch.argmax(preds_clean, 1) == targets)

            # Evaluate Backdoor
            grid_temps = (identity_grid + opt.s * noise_grid / opt.input_height) * opt.grid_rescale
            grid_temps = torch.clamp(grid_temps, -1, 1)

            ins = torch.rand(bs, opt.input_height, opt.input_height, 2).to(opt.device) * 2 - 1
            grid_temps2 = grid_temps.repeat(bs, 1, 1, 1) + ins / opt.input_height
            grid_temps2 = torch.clamp(grid_temps2, -1, 1)
            inputs_bd = F.grid_sample(inputs, grid_temps.repeat(bs, 1, 1, 1), align_corners=True)
            if opt.attack_mode == "all2one":
                targets_bd = torch.ones_like(targets) * opt.target_label
            if opt.attack_mode == "all2all":
                targets_bd = torch.remainder(targets + 1, opt.num_classes)
            preds_bd = netC(inputs_bd)
            total_bd_correct += torch.sum(torch.argmax(preds_bd, 1) == targets_bd)

            acc_clean = total_clean_correct * 100.0 / total_sample
            acc_bd = total_bd_correct * 100.0 / total_sample

            # Evaluate cross
            if opt.cross_ratio:
                inputs_cross = F.grid_sample(inputs, grid_temps2, align_corners=True)
                preds_cross = netC(inputs_cross)
                total_cross_correct += torch.sum(torch.argmax(preds_cross, 1) == targets)

                acc_cross = total_cross_correct * 100.0 / total_sample

                info_string = (
                    "Clean Acc: {:.4f} - Best: {:.4f} | Bd Acc: {:.4f} - Best: {:.4f} | Cross: {:.4f}".format(
                        acc_clean, best_clean_acc, acc_bd, best_bd_acc, acc_cross, best_cross_acc
                    )
                )
            else:
                info_string = "Clean Acc: {:.4f} - Best: {:.4f} | Bd Acc: {:.4f} - Best: {:.4f}".format(
                    acc_clean, best_clean_acc, acc_bd, best_bd_acc
                )
    progress_bar(batch_idx, len(test_dl), info_string)

    # Save checkpoint
    if acc_clean > best_clean_acc:
        print(" Saving...")
        best_clean_acc = acc_clean
        best_bd_acc = acc_bd
        if opt.cross_ratio:
            best_cross_acc = acc_cross
        else:
            best_cross_acc = torch.tensor([0])
        
        torch.save(netC.state_dict(),opt.ckpt_folder+'/model_'+str(num+1)+'.pth')

    return best_clean_acc, best_bd_acc, best_cross_acc


def main():
    opt = config.get_arguments().parse_args()

    if opt.dataset in ["mnist", "cifar10"]:
        opt.num_classes = 10
    elif opt.dataset == "gtsrb":
        opt.num_classes = 43
    elif opt.dataset == "celeba":
        opt.num_classes = 8
    else:
        raise Exception("Invalid Dataset")

    if opt.dataset == "cifar10":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "gtsrb":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "mnist":
        opt.input_height = 28
        opt.input_width = 28
        opt.input_channel = 1
    elif opt.dataset == "celeba":
        opt.input_height = 64
        opt.input_width = 64
        opt.input_channel = 3
    else:
        raise Exception("Invalid Dataset")

    # Dataset
    train_dl = get_dataloader(opt, True)
    test_dl = get_dataloader(opt, False)
    
    # Load pretrained model
    mode = opt.attack_mode
    opt.origin_folder = os.path.join(opt.checkpoints, "{}".format(opt.model))
    opt.ckpt_folder = os.path.join(opt.checkpoints, "{}_fine-tuning_{}".format(opt.model, opt.ft_ratio))
    if not os.path.exists(opt.ckpt_folder):
        os.mkdir(opt.ckpt_folder)

    print(opt.__dict__)
    
    best_performance = []
    
    for num in range(opt.model_num):

        # prepare model
        netC, optimizerC, schedulerC = get_model(opt)
        netC.load_state_dict(torch.load(opt.origin_folder+'/model_'+str(num+1)+'.pth'))
        
        print(num+1," Model: Fine-tuning!!!")
        best_clean_acc = 0.0
        best_bd_acc = 0.0
        best_cross_acc = 0.0
        epoch_current = 0

        # Prepare grid
        ins = torch.rand(1, 2, opt.k, opt.k) * 2 - 1
        ins = ins / torch.mean(torch.abs(ins))
        noise_grid = (
            F.upsample(ins, size=opt.input_height, mode="bicubic", align_corners=True)
            .permute(0, 2, 3, 1)
            .to(opt.device)
        )
        array1d = torch.linspace(-1, 1, steps=opt.input_height)
        x, y = torch.meshgrid(array1d, array1d)
        identity_grid = torch.stack((y, x), 2)[None, ...].to(opt.device)

        with open(os.path.join(opt.ckpt_folder, "opt.json"), "w+") as f:
            json.dump(opt.__dict__, f, indent=2)

        for epoch in range(opt.n_iters):
            print("Epoch {}:".format(epoch + 1))
            train(netC, optimizerC, schedulerC, train_dl, epoch, opt, num)
            best_clean_acc, best_bd_acc, best_cross_acc = eval(
                netC,
                optimizerC,
                schedulerC,
                test_dl,
                noise_grid,
                identity_grid,
                best_clean_acc,
                best_bd_acc,
                best_cross_acc,
                epoch,
                opt,
                num
            )
        best_performance.append([best_clean_acc.item(),best_bd_acc.item()])

    res = pd.DataFrame(columns=['BA','ASR'],data=best_performance)
    res.to_csv(opt.ckpt_folder+'/result.csv',index=True,encoding='utf-8')


if __name__ == "__main__":
    main()
