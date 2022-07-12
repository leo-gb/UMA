# original code: https://github.com/dyhan0920/PyramidNet-PyTorch/blob/master/train.py

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
# import pyramidnet as PYRM
import utils
import numpy as np
import torchvision.utils
from torchvision.utils import save_image
import warnings
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

import net_styleDistance
from function import adaptive_instance_normalization, coral
import torch.nn.functional as F
# from IPython import embed
# Because part of the training data is truncated image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import sys
from tqdm import tqdm
this_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(this_dir, '../models'))
from data import get_dataloader, CLSDataset, build_transform

warnings.filterwarnings("ignore")
# Check
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Cutmix PyTorch CIFAR-10, CIFAR-100 and ImageNet-1k Training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--dataset', dest='dataset', default='imagenet', type=str,
                    help='dataset (options: cifar10, cifar100, and imagenet)')
parser.add_argument('--vgg', type=str, default='./models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='./models/decoder.pth.tar')
parser.add_argument('--data_dir', type=str, default='/set/your/data/dir')
parser.add_argument('--train_file', type=str, default='leogb/causal_logs/CelebAMask-HQ-Attr19/train_attrbute_19.txt')


def main():

    global args, res, styleDistanceMatrix, numberofclass
    args = parser.parse_args()

    print('\n'+'='*100)
    print('1. init global params')
    print('='*100)
    if args.dataset == 'cifar100':
        res = torch.zeros((3, 100, 1920)).cuda()
        styleDistanceMatrix = torch.zeros((100, 100)).cuda()
    elif args.dataset == 'cifar10':
        res = torch.zeros((3, 10, 1920)).cuda()
        styleDistanceMatrix = torch.zeros((10, 10)).cuda()
    elif 'CelebAMask' in args.dataset:
        # construct the dataset
        cfg = {
            'input_size': (224, 224),
            'delimiter': '||$$||',
            'da_anno_file_list': [],
            'train_file': args.train_file,
            'dataset_name': args.train_file.split('/')[-2],
            'train_ratio': 128,
            'batch_size': 32,
            'id_nums': 2,
            'gpus': '0',
            'n_gpu': 1,
            'num_workers': 8,
            'val_num_workers': 8,
            'test_num_workers': 8,
        }
        train_transform = build_transform(cfg, split='train')
        train_dataset = CLSDataset(
            cfg['train_file'], cfg, train_transform, split='train', delimiter=cfg['delimiter'], da_anno_file_list=cfg['da_anno_file_list'])
        cfg['id_nums'] = train_dataset.num_classes
        train_loader, train_sampler = get_dataloader(cfg, train_dataset, split='train')

        res = torch.zeros((3, cfg['id_nums'], 1920)).cuda()
        styleDistanceMatrix = torch.zeros((cfg['id_nums'], cfg['id_nums'])).cuda()
    else:
        raise Exception('unknown dataset: {}'.format(args.dataset))

    print('\n'+'='*100)
    print('2. init net_styleDistance')
    print('='*100)
    global decoder, vgg, pretrained
    decoder = net_styleDistance.decoder
    vgg = net_styleDistance.vgg
    decoder.eval()
    vgg.eval()
    decoder.load_state_dict(torch.load(args.decoder))
    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:31])
    vgg.cuda()
    decoder.cuda()

    global network
    network = net_styleDistance.Net(vgg, decoder)
    network.eval()
    network = torch.nn.DataParallel(network).cuda()

    print('\n'+'='*100)
    print('3. init dataset')
    print('='*100)
    if args.dataset.startswith('cifar'):
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        if args.dataset == 'cifar100':
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100(args.data_dir+'/dataCifar100/', train=True, download=True, transform=transform_train),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            numberofclass = 100
        elif args.dataset == 'cifar10':
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(args.data_dir+'/dataCifar10/', train=True, download=True, transform=transform_train),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            numberofclass = 10
        else:
            raise Exception('unknown dataset: {}'.format(args.dataset))

    elif 'CelebAMask' in args.dataset:
        train_loader = train_loader
        numberofclass = cfg['id_nums']

    else:
        raise Exception('unknown dataset: {}'.format(args.dataset))

    print('\n'+'='*100)
    print('4. process')
    print('='*100)
    if args.dataset.startswith('cifar'):
        for i, (input, target) in enumerate(train_loader):
            # measure data loading time
            with torch.no_grad():
                input = input.cuda()
                target = target.cuda()
                u = nn.Upsample(size=(224, 224), mode='bilinear')
                content = u(input)
                mean1, std1, mean2, std2, mean3, std3, mean4, std4 = network(content)
                sv = torch.cat((mean1, std1, mean2, std2, mean3, std3, mean4, std4), 1)
                sv = sv.view(content.shape[0], -1)
                res[0, target] += sv
                res[1, target] += 1
                res[2, target] = res[0, target] / res[1, target]
        print("Total : ",res[0])
        print("Count : ",res[1])
        print("Avg : ",res[2])
        mse_loss = nn.MSELoss()
        for i in range(numberofclass):
            for j in range(numberofclass):
                styleDistanceMatrix[i, j] = mse_loss(res[2, i], res[2, j])
        torch.save(styleDistanceMatrix, './styleDistanceMatrix'+str(numberofclass)+'.pt')
        np.savetxt('./styleDistanceMatrix'+str(numberofclass), styleDistanceMatrix.cpu().numpy(), fmt='%.10e', delimiter=',')
    
    elif 'CelebAMask' in args.dataset:
        for path, input, target in tqdm(train_loader):
            # measure data loading time
            with torch.no_grad():
                input = input.cuda()
                target = target.cuda()
                u = nn.Upsample(size=(224, 224), mode='bilinear')
                content = u(input)
                mean1, std1, mean2, std2, mean3, std3, mean4, std4 = network(content)
                sv = torch.cat((mean1, std1, mean2, std2, mean3, std3, mean4, std4), 1)
                sv = sv.view(content.shape[0], -1)
                res[0, target] += sv
                res[1, target] += 1
                res[2, target] = res[0, target] / res[1, target]
        print("res.shape : ", res.shape)
        print("Total : ",res[0])
        print("Count : ",res[1])
        print("Avg : ",res[2])
        mse_loss = nn.MSELoss()
        for i in range(numberofclass):
            for j in range(numberofclass):
                styleDistanceMatrix[i, j] = mse_loss(res[2, i], res[2, j])
        # torch.save(styleDistanceMatrix, './styleDistanceMatrix'+str(numberofclass)+'.pt')
        # np.savetxt('./styleDistanceMatrix'+str(numberofclass), styleDistanceMatrix.cpu().numpy(), fmt='%.10e', delimiter=',')
        torch.save(styleDistanceMatrix, cfg['dataset_name']+'.pt')
        np.savetxt(cfg['dataset_name'], styleDistanceMatrix.cpu().numpy(), fmt='%.10e', delimiter=',')
        os.system('ossutil64 cp {}.pt oss://mvap-data/leogb/causal_logs/{}/'.format(cfg['dataset_name'], cfg['dataset_name']))
        os.system('ossutil64 cp {} oss://mvap-data/leogb/causal_logs/{}/'.format(cfg['dataset_name'], cfg['dataset_name']))

    else:
        pass

if __name__ == '__main__':
    main()
