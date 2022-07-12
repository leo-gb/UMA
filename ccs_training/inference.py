#!/usr/bin/env python3
# encoding: utf-8
"""
@author:  Yan Baoming
@contact: andy.ybm@alibaba-inc.com
"""
from __future__ import division, absolute_import
import os, sys, json, pprint, cv2
from PIL import Image
this_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(this_dir, 'models'))
sys.path.insert(0, os.path.join(this_dir, 'configs'))
from model import ImgEncoder
# from data import get_dataloader, CLSDataset, build_transform
# from utils import config_info, setup_logger, AverageMeter, LogCollector
# from lr_scheduler import WarmupMultiStepLR, WarmupCosineLR 
# from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
# import logging
import torch
from torch import nn
import torch.nn.functional as F
import argparse
import time
from oss import OssProxy, OssFile, save_txt, save_json, load_json
from io import BytesIO
from torch.utils import model_zoo
# import faiss                   # make faiss available
# from faiss import normalize_L2
# from collections import Counter
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
import psutil
from configs import oss_configs
from utils import get_world_size, get_rank
from importlib import import_module
import random
from torch.autograd import Variable
import torch.utils.data as data
from torchvision import transforms
import urllib


def parse_args():
    parser = argparse.ArgumentParser(description='Causal Coarse2fine Inference')
    parser.add_argument('--checkpoint_path', type=str, default='leogb/causal_logs/CelebAMask-HQ-Attr32/e18_r18_preFalse_step_OurX10_v1/e18_r18_preFalse_step_OurX10_v1_0001_1/ckpt_ep7.pth')
    parser.add_argument('--input_file', type=str, default='leogb/causal/CelebAMask-HQ-Attr32/editingM3L8.json')
    parser.add_argument('--save_file', type=str, default='leogb/causal/CelebAMask-HQ-Attr32/editingM3L8_new.json')
    parser.add_argument('--cfg_path', type=str, default='train_cfg')
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--random_seed', type=int, default=1)  
    parser.add_argument('--threshold', type=float, default=0.95)  
    args = parser.parse_args()
    mod = import_module(args.cfg_path)
    cfg = {
        name: value
        for name, value in mod.__dict__['cfg'].items()
        if not name.startswith('__')
    }
    cfg['checkpoint_path'] = args.checkpoint_path
    assert args.input_file != args.save_file
    cfg['input_file'] = args.input_file
    cfg['save_file'] = args.save_file
    cfg['n_gpu'] = len(args.gpus.split(','))
    cfg['batch_size'] = args.batch_size
    cfg['random_seed'] = args.random_seed
    cfg['threshold'] = args.threshold
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    pprint.pprint(cfg)
    return cfg


class TestDataset(data.Dataset):
    """docstring for TestDataset"""
    def __init__(self, input_file, transform):
        super(TestDataset, self).__init__()
        input_json = {}
        with urllib.request.urlopen(input_file) as f:
            input_json = json.load(f)
        self.keys = []
        self.images = []
        self.targets = []
        for k, v in input_json.items():
            self.keys.append(k)
            self.images.append(v[0])
            self.targets.append(v[1])
        print('len(self.keys): ', len(self.keys))
        print('len(self.images): ', len(self.images))
        print('len(self.targets): ', len(self.targets))
        self.transform = transform
        self.oss_proxy = OssProxy()

    def __getitem__(self, index):
        key = self.keys[index]
        path = self.images[index]
        target = self.targets[index]
        if os.path.isfile(path):
            img = pil_loader(path)
        else:
            img = self.oss_loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return key, path, target, img

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


def inference(cfg):
    # set random seed
    random_seed = cfg['random_seed']
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)  # cpu
    torch.cuda.manual_seed(random_seed)  # gpu
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)

    # construct the dataset
    print('=' * 10 + ' construct the dataset ' + '=' * 10)
    transform = transforms.Compose([
        transforms.Resize(cfg['input_size']),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_dataset = TestDataset(cfg['input_file'], transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg['batch_size'], sampler=None, shuffle=False, num_workers=8, pin_memory=False)

    # constuct the network
    print('=' * 10 + ' constuct the network ' + '=' * 10)
    img_encoder = ImgEncoder(cfg)
    if cfg['cuda']:
        img_encoder = img_encoder.cuda()

    print("=> loading checkpoint from {}".format(cfg['checkpoint_path']))
    if os.path.isfile(cfg['checkpoint_path']):
        checkpoint = torch.load(cfg['checkpoint_path'])
    elif 'http' in cfg['checkpoint_path']:
        checkpoint = model_zoo.load_url(cfg['checkpoint_path'])
    else:
        oss_proxy = OssProxy()
        if not oss_proxy.exists(cfg['checkpoint_path']):
            raise IOError('{} is not exists in oss'.format(cfg['checkpoint_path']))
        else:
            with OssFile(cfg['checkpoint_path']).get_bin_file() as f:
                checkpoint = torch.load(f)
    img_encoder.load_param(checkpoint['state_dict'])
    img_encoder = nn.DataParallel(img_encoder)
    img_encoder.eval()

    # process
    classify_correct = 0
    classify_total = 0
    res_dict = {}
    for i, inputs in tqdm(enumerate(test_loader), total=len(test_loader)):
        keys, paths, targets, imgs = inputs
        if cfg['cuda']:
            imgs = imgs.cuda()
            targets = targets.cuda()
        with torch.no_grad():
            logits = img_encoder(imgs)
            logits = F.softmax(logits, dim=1)
        npy_targets = targets.cpu().numpy()
        npy_logits = logits.cpu().numpy()
        npy_predicts = np.argmax(npy_logits, 1)
        npy_scores = np.max(npy_logits, 1)
        classify_correct += (npy_predicts == npy_targets).sum()
        classify_total += len(npy_targets)
        for key, path, target, predict, score in zip(keys, paths, npy_targets, npy_predicts, npy_scores):
            # if target==predict and score>cfg['threshold']:
            #     res_dict[key] = [path, predict.item()]
            if score>cfg['threshold']:
                res_dict[key] = [path, predict.item()]
                print(key, path, target, score)
    print('classification precision is {}/{} = {}%'.format(classify_correct, classify_total, 100.0*classify_correct/classify_total))

    data_buf = BytesIO()
    data_buf.write(json.dumps(res_dict, ensure_ascii=False).encode())
    oss_proxy = OssProxy()
    oss_proxy.upload_str_to_oss(data_buf.getvalue(), cfg['save_file'])
    print('save result json to : {}, len={}'.format(cfg['save_file'], len(res_dict)))


def main(cfg):
    print('start inference...')
    inference(cfg)


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)

