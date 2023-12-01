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
from models.resnet import I3Res50


def test(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    # error_data_paths = []  # 存储标签错误的数据路径

    # for idx, (data, target) in tqdm(enumerate(data_loader)):
    #     data = data.float()
    #     if args.gpu != -1:
    #         data, target = data.to(args.device), target.to(args.device)
    #     log_probs = net_g(data)
    #     # sum up batch loss
    #     test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
    #     # get the index of the max log-probability
    #     y_pred = log_probs.data.max(1, keepdim=True)[1]
    #     correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
    #     incorrect_mask = ~y_pred.eq(target.data.view_as(y_pred)).squeeze()
    #     for i in range(len(incorrect_mask)):
    #         if incorrect_mask[i]:
    #             error_data_paths = (data_loader.dataset.landmarks_frame.iloc[idx * data_loader.batch_size + i, 0])
    #             print(error_data_paths)

    # 初始化混淆矩阵
    # num_classes = len(data_loader.dataset.classes)
    confusion_matrix = np.zeros((51, 51), dtype=np.int64)

    for idx, (data, target) in tqdm(enumerate(data_loader)):
        data = data.float()
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        
        # 更新混淆矩阵
        for t, p in zip(target.view(-1), y_pred.view(-1)):
            confusion_matrix[t, p] += 1

    # 计算每个类别的准确率
    class_accuracy = {}
    for i in range(51):
        # class_name = data_loader.dataset.classes[i]
        correct_pred = confusion_matrix[i, i]
        total_pred = confusion_matrix[i, :].sum()
        class_accuracy[i] = correct_pred / total_pred

    # 打印每个类别的准确率
    for class_name, accuracy in class_accuracy.items():
        print(f'Class: {class_name}, Accuracy: {accuracy:.2%}')
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(confusion_matrix, cmap='Blues')

    # 设置坐标轴标签
    ax.set_xticks(np.arange(51))
    ax.set_yticks(np.arange(51))
    ax.xaxis.set_tick_params(labeltop=True, labelbottom=False)

    # 在热图中显示数值
    for i in range(51):
        for j in range(51):
            text = ax.text(j, i, confusion_matrix[i, j],
                        ha="center", va="center", color="w")

    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax)

    # 设置图像标题和标签
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    # 自动调整布局
    plt.tight_layout()

    # 显示图像
    plt.savefig('./save/ConfusionMatrix_train.png')

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    return accuracy, test_loss

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print('1.device',args.device)
    # load dataset and split users
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip(),
                                           videotransforms.TransposeIMG()
    ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224),videotransforms.TransposeIMG()])
    print("2.load dataset")
    if args.dataset == 'bupt':
        val_dataset = videodataset(args.train_split, os.path.join(args.video_path, 'test'), transform=test_transforms)
    elif args.dataset == 'hmdb':
        val_dataset = videodataset(args.train_split, args.video_path, transform=test_transforms) # split
    else:
        exit('Error: unrecognized dataset')
    test_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)
    # build model
    print("4.build model")
    net_glob = I3Res50(num_classes = 51)
    net_glob.load_state_dict(torch.load('logs/baseline_59.pth'))
    net_glob = net_glob.to(args.device)
    net_glob.eval()

    acc_test, loss_test = test(net_glob, val_dataset, args)

    print(acc_test)


