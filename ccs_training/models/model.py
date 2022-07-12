#!/usr/bin/env python3
# encoding: utf-8
"""
@author:  Yan Baoming
@contact: andy.ybm@alibaba-inc.com
"""

from torch import nn
import torch
from resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2
from resnet import weights_init_kaiming
from moex_resnet import moex_resnet18, moex_resnet34, moex_resnet50, moex_resnet101, moex_resnet152, pono_resnext50_32x4d, pono_resnext101_32x8d, wide_moex_resnet50_2, wide_moex_resnet101_2
from resnest import resnest50, resnest101
from resnext_ibn_a import resnext101_ibn_a
from mobilenet_v2 import mobilenet_v2_1d0, mobilenet_v2_0d5
from effcientnet import efficientnet_b3, efficientnet_b4, efficientnet_b5
from pooling import GeneralizedMeanPoolingP, SelfAttenPooling
import pickle
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import os
from oss import OssProxy, OssFile


net_dict = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnext50_32x4d': resnext50_32x4d,
    'resnext101_32x8d': resnext101_32x8d,
    'wide_resnet50_2': wide_resnet50_2,
    'wide_resnet101_2': wide_resnet101_2,
    'resnest50': resnest50,
    'resnest101': resnest101,
    'resnext101_ibn_a': resnext101_ibn_a,
    'mobilenet_v2_1d0' : mobilenet_v2_1d0,
    'mobilenet_v2_0d5' : mobilenet_v2_0d5,
    'efficientnet_b3': efficientnet_b3,
    'efficientnet_b4': efficientnet_b4,
    'efficientnet_b5': efficientnet_b5,
    'moex_resnet18': moex_resnet18,
    'moex_resnet34': moex_resnet34,
    'moex_resnet50': moex_resnet50,
    'moex_resnet101': moex_resnet101,
    'moex_resnet152': moex_resnet152,
    'pono_resnext50_32x4d': pono_resnext50_32x4d,
    'pono_resnext101_32x8d': pono_resnext101_32x8d,
    'wide_moex_resnet50_2': wide_moex_resnet50_2,
    'wide_moex_resnet101_2': wide_moex_resnet101_2
}


def l2norm(X, dim=1, return_norm=False):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt()
    norm = torch.clamp(norm, min=1e-10)
    X = torch.div(X, norm)
    if return_norm:
        return X, norm
    return X


class ImgEncoder(nn.Module):
    """docstring for ImgEncoder"""
    def __init__(self, cfg):
        super(ImgEncoder, self).__init__()
        self.cfg = cfg
        self.model_name = self.cfg['model_name']
        self.pretrained = self.cfg['pretrained']
        print('{} backbone'.format(self.model_name))
        assert self.model_name in net_dict.keys()
        self.net = net_dict[self.model_name](pretrained=self.pretrained, cfg=cfg)
        if self.cfg['gem']:
            print('use gem pooling')
            self.pooling = GeneralizedMeanPoolingP()
        else:
            print('use avarge pooling')
            self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        self.id_nums = self.cfg['id_nums']
        self.classifier = nn.Linear(self.cfg['feat_dims'], self.id_nums)

    def forward(self, x, label=None, is_train=False, swap_index=None):

        if self.cfg.get('Manifold_Mixup', False) and is_train:
            base_feats, target_reweighted = self.net(x, target=label, is_train=is_train)
        elif self.cfg.get('MoEx', False):
            base_feats = self.net(x, swap_index=swap_index)
        else:
            base_feats = self.net(x)

        pooled_feats = self.pooling(base_feats).view(base_feats.shape[0], -1)
        logits = self.classifier(pooled_feats)
        if self.cfg.get('Manifold_Mixup', False) and is_train:
            return logits, target_reweighted
        else:
            return logits

    def load_param(self, model_path):
        if isinstance(model_path, str):
            print("=> loading imgencoder checkpoint {}".format(model_path))
            param_dict = torch.load(model_path)['state_dict']
        else:
            print("=> loading imgencoder from dict")
            param_dict = model_path
        for i in param_dict:
            if i not in self.state_dict().keys():
                print('skip {}'.format(i))
                continue
            if self.state_dict()[i].shape != param_dict[i].shape:
                print('skip {}, shape dismatch {} vs {}'.format(i, self.state_dict()[i].shape, param_dict[i].shape))
                continue
            self.state_dict()[i].copy_(param_dict[i])


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
        'model_name': 'resnet18',
        'pretrained': False,
        'gem': False,
        'feat_dims': 512,
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

    x = torch.rand(1, 3, 256, 256)
    label = torch.tensor([[1], [2]])
    img_encoder = ImgEncoder(cfg)
    # print(img_encoder.state_dict().keys())
    # img_encoder = nn.SyncBatchNorm.convert_sync_batchnorm(img_encoder)
    print(img_encoder)
    print(img_encoder(x).shape)
    # base_feats, pooled_feats, logits = img_encoder(x, label)
    # print('img encoder', base_feats.shape, pooled_feats.shape, logits.shape)

    # img_feat_encoder = ImgfeatEncoder(cfg)
    # x = torch.rand(2, 2048, 8, 8)
    # print('img_feat_encoder', img_feat_encoder(x).shape)

    # text_encoder = TextEncoder(cfg).cuda()
    # text_input = (torch.rand(2, 20) * 10).long().cuda()
    # lengths = [8, 5]
    # print(text_input, lengths)
    # print('text encoder', text_encoder(text_input, lengths).shape)


