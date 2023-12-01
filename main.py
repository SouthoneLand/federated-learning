#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import os
from datetime import datetime
from tqdm import tqdm

from utils.VideoDataset import videodataset
from utils.VidpicDataset import vidpicdataset
from utils import videotransforms
from utils.videosampling import iid, noniid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Fed import FedAvg
from models.test import test_img
from models.i3d import InceptionI3d
from models.I3D import I3D
from models.resnet import i3_res50

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print('1.device',args.device)

    save_model = args.save_path
    if not os.path.exists(save_model):
        os.makedirs(save_model)
    current_time = datetime.now()
    time_string = current_time.strftime("%Y%m%d_%H%M")
    with open(save_model + time_string + '_log.txt', 'a') as file:
        file.write(f"DataIID: {args.iid}\nTotal epoch: {args.epochs}\nClients: {args.num_users}\nFraction: {args.frac}\nLocal epoch: {args.local_ep}\nlr: {args.lr}\tmomentum:{args.momentum}\n")

    # load dataset and split users
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip(),
                                           videotransforms.TransposeIMG()
    ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224),videotransforms.TransposeIMG()])
    print("2.load dataset")
    if args.dataset == 'bupt':
        dataset_train = vidpicdataset(args.train_split, os.path.join(args.video_path, 'train'), transform=train_transforms)
        val_dataset = vidpicdataset(args.val_split, os.path.join(args.video_path, 'test'), transform=test_transforms)
    elif args.dataset == 'hmdb':
        dataset_train = vidpicdataset(args.train_split, args.video_path, transform=train_transforms)
        val_dataset = vidpicdataset(args.val_split, args.video_path, transform=test_transforms)
        if args.iid:
            print("3.IID")
            dict_users = iid(dataset_train, args.num_users)
        else:
            print("3.nonIID")
            dict_users = noniid(dataset_train, args.num_users)
    else:
        exit('Error: unrecognized dataset')
    train_loader = DataLoader(dataset_train, batch_size=16, shuffle=True)
    test_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
    # build model
    print("4.build model")
    net_glob = i3_res50(400)
    # net_glob.load_state_dict(torch.load(args.pre_path))
    net_glob.replace_logits(num_classes=args.num_classes)
    net_glob = net_glob.to(args.device)
    
    optimizer = optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=0.000001)
    lr_sched = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.9)

    # training
    print("5.training...")
    loss_train = []
    acc_best = 0.
    net_glob.train()
    for epoch in range(args.epochs):
        batch_loss = []
        correct = 0
        test_correct = 0
        for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
            data = data.float()
            data, target = data.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            output = net_glob(data)
            y_pred = output.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, target)
            # loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        loss_avg = sum(batch_loss)/len(batch_loss)
        acc_train = 100.00 * correct / len(train_loader.dataset)
        print('Round {:3d}, Average loss {:.3f}, Training accuracy: {:.2f}\n'.format(epoch, loss_avg, acc_train))
        with open(save_model + time_string + '_log.txt', 'a') as file:
            file.write('Round {:3d}, Average loss {:.3f}, Training accuracy: {:.2f}\n'.format(epoch, loss_avg, acc_train))
        loss_train.append(loss_avg)
        if (epoch+1)%5 == 0:
            net_glob.eval()
            for batch_idx, (data, target) in tqdm(enumerate(test_loader)):
                data = data.float()
                data, target = data.to(args.device), target.to(args.device)
                output = net_glob(data)
                y_pred = output.data.max(1, keepdim=True)[1]
                test_correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
            acc_test = 100.00 * test_correct / len(test_loader.dataset)
            if acc_best < acc_test:
                torch.save(net_glob.state_dict(), save_model + time_string + '_best.pth')
                acc_best = acc_test
                print('Best test accuracy: {:.2f}\n'.format(acc_best))
                with open(save_model + time_string + '_log.txt', 'a') as file:
                    file.write('Best test accuracy: {:.2f}\n'.format(acc_best))
            else:
                print('Testing accuracy: {:.2f}\n'.format(acc_test))
                with open(save_model + time_string + '_log.txt', 'a') as file:
                    file.write('Testing accuracy: {:.2f}\n'.format(acc_test))
            net_glob.train()

    # testing
    if args.epochs%5 != 0:
        print("testing...")
        net_glob.eval()
        acc_train, loss_train = test_img(net_glob, dataset_train, args)
        acc_test, loss_test = test_img(net_glob, val_dataset, args)
        print("Training accuracy: {:.2f}".format(acc_train))
        print("Testing accuracy: {:.2f}".format(acc_test))
        if acc_best < acc_test:
            torch.save(net_glob.state_dict(), save_model + time_string + '_best.pth')
            acc_best = acc_test
        with open(save_model + time_string + '_log.txt', 'a') as file:
            file.write('Average loss {:.3f}, Training accuracy: {:.2f}, Testing accuracy: {:.2f}, Best test accuracy: {:.2f}\n'.format(loss_avg, acc_train, acc_test, acc_best))
    print("Best test accuracy: {:.2f}".format(acc_best))
    with open(save_model + time_string + '_log.txt', 'a') as file:
        file.write("Best test accuracy: {:.2f}".format(acc_best))



