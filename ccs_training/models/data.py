#!/usr/bin/env python3
# encoding: utf-8
"""
@author:  Yan Baoming
@contact: andy.ybm@alibaba-inc.com
"""
from __future__ import division, absolute_import
import os
from PIL import Image
import torch.utils.data as data
import json
import copy
import numpy as np
import random
from collections import defaultdict
import torch
from torch.utils.data.sampler import Sampler
from autoaugment import CIFAR10Policy
from autoaugment_v2 import AutoAugment
from torchvision import transforms
from randaugment import RandAugment
import pickle
from oss import OssProxy, OssFile
from io import BytesIO
import cv2
from random_blur import RandomBlur
import torchvision.transforms.functional as F
import itertools
import time
import oss2

buffer_state = True
try:
    from dmls.dmls import Client
except Exception:
    buffer_state = False
    print('warnning: no dmls installed !!!')

import imghdr, imageio, string
import multiprocessing as mp
from threading import Thread
from multiprocessing.pool import ThreadPool
import urllib
import urllib.request
from urllib.error import URLError, HTTPError
from urllib.parse import quote
from datetime import datetime
def download_image_mp(img_url):
    img = None
    try:
        img_url = quote(img_url, safe=string.printable)
        req = urllib.request.urlopen(img_url)
    except HTTPError as e:
        # print('Error code: ', e.code)
        pass
    except URLError as e:
        # print('Reason: ', e.reason)
        pass
    else:
        reqdata = req.read()
        imgformat = imghdr.what('', h=reqdata)
        if imgformat in ['gif']:
            save_path='./data/'
            os.makedirs(os.path.join(save_path, 'gif'), exist_ok=True)
            fname = os.path.join(save_path, 'gif', 'tmp_{}.gif'.format(datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S_%f')))
            # print('GIF: ', img_url, fname)
            with open(fname, 'wb+') as f:
                f.write(bytearray(reqdata))
            gif = imageio.mimread(fname)
            imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in gif]
            img = imgs[0]
            if os.path.exists(fname):
                os.remove(fname)
        else:
            arr = np.asarray(bytearray(reqdata), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        # img = img.resize((256, 256))
    return img


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def crop_img(img, bbox, mode='val'):
    width, height = img.size
    # if mode == 'train':
    #     ratio = 0.1 + ((random.random() - 0.5) * 0.02)  # [0.09, 0.11]
    # else:
    #     ratio = 0.1
    ratio = 0
    x1, y1, x2, y2 = bbox
    x_margin = int((x2 - x1) * ratio)
    y_margin = int((y2 - y1) * ratio)
    x1 = max(0, x1 - x_margin)
    y1 = max(0, y1 - y_margin)

    x2 = min(x2 + x_margin, width)
    y2 = min(y2 + y_margin, height)
    img = img.crop((x1, y1, x2, y2))
    return img


class CLSDataset(data.Dataset):
    """docstring for CLSDataset"""
    def __init__(self, anno_file, cfg, transform, split='train', delimiter='\t', da_anno_file_list=[]):
        super(CLSDataset, self).__init__()
        self.anno_file = anno_file
        self.cfg = cfg
        self.oss_proxy = OssProxy()
        self.split = split
        if os.path.isfile(anno_file):
            with open(anno_file, 'r') as f:
                self.anno = f.readlines()
        else:
            if 'http' in anno_file:
                with urllib.request.urlopen(anno_file) as f:
                    self.anno = [str(line.decode('utf-8')).strip('\n') for line in f]
            else:
                with OssFile(anno_file).get_str_file() as f:
                    self.anno = f.readlines()

        self.images = [tt.strip().split(delimiter)[-2] for tt in self.anno]
        self.targets = [int(tt.strip().split(delimiter)[-1]) for tt in self.anno]
        self.images_neg = [tt.strip().split(delimiter)[-2] for tt in self.anno if int(tt.strip().split(delimiter)[-1])==0]
        print('> ALL - len(self.images): {}'.format(len(self.images)))
        print('> ALL - len(self.targets): {}'.format(len(self.targets)))
        print('> ALL - len(self.images_neg): {}'.format(len(self.images_neg)))

        if split == 'train':
            train_numbers = round(min(cfg['train_ratio'] * cfg['batch_size'], len(self.images)))
            self.images = self.images[:train_numbers]
            self.targets = self.targets[:train_numbers]
            print('> len(self.images): {}'.format(len(self.images)))
            print('> len(self.targets): {}'.format(len(self.targets)))

            # e4e image data augmentation
            # da_anno_file_list = [
            #     'leogb/causal/CelebAMask-HQ/CelebAMask-HQ-9_editingM1L8.json',      #同类，单层交换
            #     'leogb/causal/CelebAMask-HQ/CelebAMask-HQ-9_editingM2Ltop8.json',   #同类，多层交换
            #     'leogb/causal/CelebAMask-HQ/CelebAMask-HQ-9_editingM3L4.json',      #不同类，单层交换
            #     'leogb/causal/CelebAMask-HQ/CelebAMask-HQ-9_editingM4Ltop4.json',   #不同类，多层交换
            # ]
            da_images = []
            da_targets = []
            da_images_count_by_m1m2 = 0
            da_images_count_by_m3m4 = 0
            for da_anno_file in da_anno_file_list:
                da_anno = {}
                with open(da_anno_file) as f:
                    da_anno = json.load(f)
                for imgname in self.images:
                    key = imgname.split('/')[-1]
                    if key not in da_anno.keys():
                        continue
                    da_i, da_t = da_anno[key]
                    #if int(da_t)==1:    #只生成扩充正样本
                    if 1:
                        da_images.append(da_i)
                        da_targets.append(int(da_t))
                    else:
                        da_images.append(random.choice(self.images_neg))
                        da_targets.append(0)

                    if 'editingM3' in da_anno_file or 'editingM4' in da_anno_file:
                        print(da_i, da_t)
                        da_images_count_by_m3m4 += 1
                    else:
                        da_images_count_by_m1m2 += 1

            self.images += da_images
            self.targets += da_targets
            print('> DA len(self.images): {}'.format(len(self.images)))
            print('> DA len(self.targets): {}'.format(len(self.targets)))
            print('> da_images_count_by_m1m2: {}'.format(da_images_count_by_m1m2))
            print('> da_images_count_by_m3m4: {}'.format(da_images_count_by_m3m4))

        self.num_classes = 2
        self.transform = transform
        print('{} images number: {}, num classes: {}'.format(split, len(self.images), self.num_classes))

        # if cfg.get('load_data_to_memory', False):
        #     print('load all image to memory...')
        #     st = time.time()
        #     # 并行下载评价图
        #     # -------------------
        #     # pool = mp.Pool(min(32, len(self.images)))
        #     # self.images_data = pool.map(download_image_mp, self.images)
        #     # pool.close()
        #     # -------------------
        #     threadpool = ThreadPool(min(32, len(self.images)))
        #     self.images_data = threadpool.starmap(download_image_mp, [(url,) for url in self.images])
        #     threadpool.close()
        #     threadpool.join()
        #     assert len(self.images) == len(self.images_data)

        #     temp_images = []
        #     temp_targets = []
        #     temp_images_data = []
        #     for index, img in enumerate(self.images_data):
        #         if img is None:
        #             pass
        #         else:
        #             temp_images.append(self.images[index])
        #             temp_targets.append(self.targets[index])
        #             temp_images_data.append(self.images_data[index])
        #     self.images = temp_images
        #     self.targets = temp_targets
        #     self.images_data = temp_images_data
        #     print('> ALL - len(self.images): {}'.format(len(self.images)))
        #     print('> ALL - len(self.targets): {}'.format(len(self.targets)))
        #     print('> ALL - len(self.images_data): {}'.format(len(self.images_data)))

        #     et = time.time()
        #     print('load {} images to memory, time elapsed : {:>.4f}s'.format(len(self.images), et-st))
        

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.images[index]
        target = self.targets[index]

        # if self.cfg.get('load_data_to_memory', False):
        #     img = self.images_data[index]
        #     if self.transform is not None:
        #         img = self.transform(img)
        #     return path, img, target

        # if self.split == 'train':
        #     ref_index = random.choice(range(len(self.images)))
        #     ref_path = self.images[ref_index]
        #     ref_target = self.targets[ref_index]
        #     if target==ref_target:
        #         s_name = path.split('/')[-1].split('.')[0]
        #         r_name = ref_path.split('/')[-1].split('.')[0]
        #         da_type = random.choice(['S8', 'M4', 'M8', 'M12'])
        #         path = 'leogb/causal/CelebAMask-HQ-Attr32/editingM5/{}_{}_{}_{}.jpg'.format(s_name, r_name, da_type, target)
        #         print(path)

        if os.path.isfile(path):
            img = pil_loader(path)
        elif path[:4] == 'http':
            img = download_image_mp(path)
        else:
            img = self.oss_loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return path, img, target

    def __len__(self):
        return len(self.images)

    def oss_loader(self, img_path):
        img = None
        for _ in range(10):  # try 10 times
            try:
                data = self.oss_proxy.download_to_bytes(img_path)
                temp_buffer = BytesIO()
                temp_buffer.write(data)
                temp_buffer.seek(0)
                img = np.fromstring(temp_buffer.getvalue(), np.uint8)
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                temp_buffer.close()
            except Exception as err:
                print('load image error:', img_path, err)
            if img is not None:
                break
        return img


_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


class Resize(object):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        tmp_size = random.randint(round(self.size * 1.1), round(self.size * 1.2))
        # print('tmp_size', tmp_size)
        return F.resize(img, self.size, self.interpolation)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


def build_transform(cfg, split='train'):
    print('get transforms')
    if not isinstance(cfg['input_size'], (tuple, list)):
        cfg['input_size'] = (cfg['input_size'], cfg['input_size'])
    if split == 'train':
        transforms_list = []

        if cfg.get('random_crop', False):
            transforms_list.append(Resize(min(cfg['input_size'])))
            transforms_list.append(transforms.RandomCrop(cfg['input_size']))
        else:
            transforms_list.append(transforms.Resize(cfg['input_size']))

        if cfg.get('AutoAugment', False):
            transforms_list.append(AutoAugment())
            # transforms_list.append(CIFAR10Policy(fillcolor=(255, 255, 255)))

        if cfg.get('RandAugment', False):
            transforms_list.append(RandAugment(2, 14))

        transforms_list.append(transforms.RandomHorizontalFlip())

        if cfg.get('random_distortion', False):
            transforms_list.append(transforms.Ran)

        if  cfg.get('random_rotate', False):
            import Augmentor
            p = Augmentor.Pipeline()
            if cfg.get('random_distortion', False):
                p.random_distortion(probability=0.5, grid_width=3, grid_height=3, magnitude=5)
            # if cfg.get('random_rotate', False):
            #     p.rotate(probability=0.5, max_left_rotation=10, max_right_rotation=10)
            transforms_list.append(p.torch_transform())

        if cfg.get('random_blur', False):
            transforms_list.append(RandomBlur())

        transforms_list.append(transforms.ToTensor())

        if cfg.get('data_normalize', True):
            transforms_list.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        
        if cfg.get('random_erasing', False):
            transforms_list.append(transforms.RandomErasing(scale=(0.01, 0.06), value='random'))

        if cfg.get('To_pil', False):
            transforms_list.append(transforms.ToPILImage())

        print(transforms_list)
        transform = transforms.Compose(transforms_list)
    elif split == 'valid' or split == 'test':
        transform = transforms.Compose([
            transforms.Resize(cfg['input_size']),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transform


def get_dataloader(cfg, dataset, split='train'):
    gpu_nums = 1
    if cfg.get('distributed', False):
        gpu_nums = cfg['n_gpu']
    if split == 'train':
        print('use RandomSampler')
        if cfg.get('distributed', False):
            data_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
        else:
            data_sampler = torch.utils.data.RandomSampler(dataset)
    else:
        if cfg.get('distributed', False):
            data_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
        else:
            data_sampler = None

    if split == 'train':
        dataloader = torch.utils.data.DataLoader(dataset,
            batch_size=cfg['batch_size']//gpu_nums,
            sampler=data_sampler,
            num_workers=cfg['num_workers']//gpu_nums,
            pin_memory=False)
    elif split == 'valid':
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg['val_batch_size']//gpu_nums,
            sampler=data_sampler,
            shuffle=False,
            num_workers=cfg['val_num_workers']//gpu_nums,
            pin_memory=False)
    elif split == 'test':
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg['test_batch_size']//gpu_nums,
            sampler=data_sampler,
            shuffle=False,
            num_workers=cfg['val_num_workers']//gpu_nums,
            pin_memory=False)
    return dataloader, data_sampler


if __name__ == '__main__':
    cfg = {
        # data
        'train_file': 'pregnent_data/train_label.txt',
        'valid_file': 'pregnent_data/valid_label.txt',
        'input_size': (256, 256),
        'num_workers': 0,
        'batch_size': 32,
        'val_batch_size': 32,
        'test_batch_size': 32,

        # model
        'mode': 'train',
        'model_name': 'resnet50',
        'pretrained': True,
        'gem': True,
        'classifier_name': 'CircleLoss',
        'feat_dims': 2048,
        'id_nums': 2,

        # optimizer
        'margin': 0.3,
        'ce_weight': 0.1,
        'triplet_weight': 0.9,
        'weight_decay': 0.0005,
        'warmup_iters': 10,
        'learning_rate': 7e-4,
        'lr_scheduler': 'lr',
        'milestones': [35, 70],
        'gamma': 0.1,
        'warmup_iters': 10,
        'max_epochs': 100,
        'cuda': True,

        # save path
        'log_dir': 'logs_platform',
        'snap': 'resnet50_256x256',
        'print_interval': 100,
        'valid_interval': 1000

    }
    transform = build_transform(cfg)
    cls_dataset = CLSDataset(cfg['train_file'], cfg, transform, delimiter='$$||$$')
    data_loader, sampler = get_dataloader(cfg, cls_dataset)
    for epoch in range(3):
        print(epoch)
        for path, img, target in data_loader:
            print(path[0], target)
            print(len(path), img.shape, target.shape)
