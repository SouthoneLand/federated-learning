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
import os
from datetime import datetime
from scipy import signal
import torch.optim.lr_scheduler as lr_scheduler

# from utils.VideoDataset import videodataset
from utils.VidpicDataset import vidpicdataset
from utils import videotransforms
from utils.videosampling import iid, noniid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Fed import FedAvg, FedNova
from models.test import test_img
from models.i3d import InceptionI3d
from models.I3D import I3D
from models.resnet import i3_res50

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print('1.device',args.device)
    
    assert args.method in ['fedavg', 'fedprox', 'fedl2', 'fedl1', 'moon', 'fednova']

    save_model = args.save_path
    if not os.path.exists(save_model):
        os.makedirs(save_model)
    current_time = datetime.now()
    time_string = current_time.strftime("%Y%m%d_%H%M")
    with open(save_model + time_string + '_log.txt', 'a') as file:
        file.write(f"DataIID: {args.iid}\nTotal epoch: {args.epochs}\n" \
        f"Clients: {args.num_users}\nFraction: {args.frac}\n" \
        f"Local epoch: {args.local_ep}\nlr: {args.lr}\tmomentum:{args.momentum}\n" \
        f"Method: {args.method}\n")

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

    # build model
    print("4.build model")
    net_glob = i3_res50(400)
    # net_glob.load_state_dict(torch.load(args.pre_path))
    net_glob.replace_logits(num_classes=args.num_classes)
    net_glob = net_glob.to(args.device)

    optimizer = torch.optim.SGD(net_glob.parameters(), lr=args.lr)
    lr_sched = lr_scheduler.MultiStepLR(optimizer, milestones=[150,300], gamma=0.9)

    # copy weights
    w_glob = net_glob.state_dict()
    if args.method == 'moon':
        net_locals = [copy.deepcopy(net_glob) for i in range(args.num_users)]

    # cyclic server learning rate
    # time = np.arange(0, 2, 0.01)
    # freq = args.freq
    # max_clr = 0
    # clr_cycle = args.amp * signal.sawtooth(2 * np.pi * freq * time)
    # clr = 1.0 - clr_cycle

    # training
    print("5.training...")
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    acc_best = 0.
    
    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    for iter in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        lr = lr_sched.get_lr()[0]
        lr_sched.step()
        if args.method == 'fednova':
            local_len = []
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], lr=lr)
            if args.method == 'moon':
                w, loss = local.train(previous_net=copy.deepcopy(net_locals[idx]).to(args.device), net=copy.deepcopy(net_glob).to(args.device))
                net_locals[idx].load_state_dict(w)
            else:
                w, loss = local.train(previous_net=None, net=copy.deepcopy(net_glob).to(args.device))
            if args.method == 'fednova':
                local_len.append(local.data_num)
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        if args.method == 'fednova':
            w_glob = FedNova(w_glob, w_locals, args, local_len)
        else:
            w_glob = FedAvg(w_locals, args)
        # elif args.fed is 'moon':
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
        # testing
        # print(iter)
        if (iter+1)%5 == 0:
            net_glob.eval()
            acc_train, _ = test_img(net_glob, dataset_train, args)
            acc_test, _ = test_img(net_glob, val_dataset, args)
            print("Training accuracy: {:.2f}".format(acc_train))
            print("Testing accuracy: {:.2f}".format(acc_test))
            with open(save_model + time_string + '_log.txt', 'a') as file:
                file.write('Round {:3d}, Average loss {:.3f}, Training accuracy: {:.2f}, Testing accuracy: {:.2f}\n'.format(iter, loss_avg, acc_train, acc_test))
            # with open(save_model + time_string + '_log.txt', 'a') as file:
            #     file.write('Round {:3d}, Average loss {:.3f}, Testing accuracy: {:.2f}\n'.format(iter, loss_avg, acc_test))
            if acc_best < acc_test:
                torch.save(net_glob.state_dict(), save_model + time_string + '_best.pth')
                acc_best = acc_test
            net_glob.train()
    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.method, args.epochs, args.frac, args.iid))

    # testing
    if args.epochs%10 != 0:
        net_glob.eval()
        acc_train, loss_train = test_img(net_glob, dataset_train, args)
        acc_test, loss_test = test_img(net_glob, val_dataset, args)
        print("Training accuracy: {:.2f}".format(acc_train))
        print("Testing accuracy: {:.2f}".format(acc_test))
        if acc_best < acc_test:
            torch.save(net_glob.state_dict(), save_model + time_string + '_best.pth')
            acc_best = acc_test
        with open(save_model + time_string + '_log.txt', 'a') as file:
            file.write('Round {:3d}, Average loss {:.3f}, Training accuracy: {:.2f}, Testing accuracy: {:.2f}, Best test accuracy: {:.2f}\n'.format(iter, loss_avg, acc_train, acc_test, acc_best))
    print("Best test accuracy: {:.2f}".format(acc_best))
    with open(save_model + time_string + '_log.txt', 'a') as file:
        file.write("Best test accuracy: {:.2f}".format(acc_best))
