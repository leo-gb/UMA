#!/usr/bin/env python3
# encoding: utf-8
"""
@author:  Yan Baoming
@contact: andy.ybm@alibaba-inc.com
"""
from __future__ import division, absolute_import
import os
import sys
this_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(this_dir, 'models'))
sys.path.insert(0, os.path.join(this_dir, 'configs'))
from model import ImgEncoder
from data import get_dataloader, CLSDataset, build_transform
from utils import config_info, setup_logger, AverageMeter, LogCollector
from lr_scheduler import WarmupMultiStepLR, WarmupCosineLR 
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import logging
import torch
from torch import nn
import torch.nn.functional as F
import argparse
import time
from oss import OssProxy, OssFile, save_txt, save_json, load_json
from io import BytesIO
from torch.utils import model_zoo
import faiss                   # make faiss available
from faiss import normalize_L2
from collections import Counter
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
import psutil
from configs import oss_configs
from utils import get_world_size, get_rank
from importlib import import_module
import random
from torch.autograd import Variable
from mixup import mixup_data, mixup_criterion
from cutmix import cutmix_data, cutmix_criterion
try:
    from apex import amp
except Exception as e:
    print('warnning: apex is not installed !!!')


from PIL import Image
def tensor2im(var):
    # var shape: (3, H, W)
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    # print('111', var.max(), var.min())
    var = var*np.array([0.229,0.224,0.225])
    var = var+np.array([0.485,0.456,0.406])
    # var[0,:,:] = var[0,:,:]*0.229+0.485
    # var[1,:,:] = var[0,:,:]*0.224+0.456
    # var[2,:,:] = var[0,:,:]*0.225+0.406
    # print('222', var.max(), var.min())

    # var = ((var + 1) / 2)
    # print('333', var.max(), var.min())

    var[var < 0] = 0
    var[var > 1] = 1
    # print('444', var.max(), var.min())

    var = var * 255
    # print('555', var.max(), var.min())

    return Image.fromarray(var.astype('uint8'))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Causal Coarse2fine Training')
    parser.add_argument('--task_name', type=str, default='baseline_r50')
    parser.add_argument('--round', type=str, default='1')
    parser.add_argument('--log_dir', type=str, default='bmyan/exps/coarse2fine/CelebAMask-HQ/baseline/')
    parser.add_argument('--train_file', type=str, default='bmyan/dataset/CelebAMask-HQ/train_annos/train_attrbute_9.txt')
    parser.add_argument('--valid_file', type=str, default='bmyan/dataset/CelebAMask-HQ/train_annos/valid_attrbute_9.txt')
    parser.add_argument('--load_data_to_memory', action='store_true', help='if load all images data to memory')

    parser.add_argument('--delimiter', type=str, default='||$$||')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--distributed', action='store_true', help='if use distributed training')
    parser.add_argument('--cfg_path', type=str, default='train_cfg')
    parser.add_argument('--train_ratio', type=int, default=10000)
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--gpu0_bsz', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--random_seed', type=int, default=1)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--norm', type=str, default='BN')
    parser.add_argument('--valid', action='store_true', help='if use validation')
    parser.add_argument('--apex', action='store_true', help='if use apex training')
    parser.add_argument('--wait_pid', type=int, default=-1, help='exp name')
    parser.add_argument('--empty_cache', action='store_true', help='if use cuda empty_cache')
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--OverwriteDAConfig', action='store_true', help='if use OverwriteDAConfig')
    parser.add_argument('--Mixup', action='store_true', help='if use Mixup')
    parser.add_argument('--Cutmix', action='store_true', help='if use Cutmix')
    parser.add_argument('--AutoAugment', action='store_true', help='if use AutoAugment')
    parser.add_argument('--Manifold_Mixup', action='store_true', help='if use Manifold_Mixup')
    parser.add_argument('--random_erasing', action='store_true', help='if use random_erasing')
    parser.add_argument('--RandAugment', action='store_true', help='if use RandAugment')
    parser.add_argument('--MoEx', action='store_true', help='if use MoEx')
    parser.add_argument('--StyleMix', action='store_true', help='if use StyleMix')
    parser.add_argument('--StyleMix_method', type=str, default='StyleCutMix_Auto_Gamma')
    parser.add_argument('--UMAx6', action='store_true', help='if use UMAx6')
    parser.add_argument('--UMAx10', action='store_true', help='if use UMAx10')
    parser.add_argument('--UMAx3', action='store_true', help='if use UMAx3')
    parser.add_argument('--UMAx5', action='store_true', help='if use UMAx5')
    parser.add_argument('--UMAx10_PLv1', action='store_true', help='if use UMAx10_PLv1')
    parser.add_argument('--UMAx10_PLv2', action='store_true', help='if use UMAx10_PLv2')
    parser.add_argument('--UMAx10_PLv3', action='store_true', help='if use UMAx10_PLv3')
    parser.add_argument('--UMAx10_PLv4', action='store_true', help='if use UMAx10_PLv4')

    args = parser.parse_args()
    mod = import_module(args.cfg_path)
    cfg = {
        name: value
        for name, value in mod.__dict__['cfg'].items()
        if not name.startswith('__')
    }
    cfg['mode'] = args.mode
    cfg['resume'] = args.resume
    cfg['pretrained_model'] = args.pretrained_model
    cfg['log_dir'] = args.log_dir
    cfg['snap'] = '{}_{:>04}_{}'.format(args.task_name, args.train_ratio, args.round)
    cfg['batch_size'] = args.batch_size
    cfg['delimiter'] = args.delimiter
    cfg['train_ratio'] = args.train_ratio

    cfg['train_file'] = args.train_file
    cfg['valid_file'] = args.valid_file
    cfg['load_data_to_memory'] = args.load_data_to_memory
    cfg['valid'] = args.valid
    cfg['n_gpu'] = len(args.gpus.split(','))
    cfg['apex'] = args.apex
    cfg['norm'] = args.norm
    cfg['empty_cache'] = args.empty_cache
    cfg['distributed'] = args.distributed
    cfg['random_seed'] = args.random_seed
    # torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    cfg['local_rank'] = args.local_rank
    # cfg['world_size'] = get_world_size()
    # cfg['rank'] = get_rank()

    if args.OverwriteDAConfig:
        print('='*50)
        print('Overwriting Data Augmentation Config...')
        print('='*50)
        cfg['Mixup'] = args.Mixup
        cfg['Cutmix'] = args.Cutmix
        cfg['AutoAugment'] = args.AutoAugment
        cfg['Manifold_Mixup'] = args.Manifold_Mixup
        cfg['random_erasing'] = args.random_erasing
        cfg['RandAugment'] = args.RandAugment
        cfg['MoEx'] = args.MoEx
        cfg['model_name'] = 'moex_resnet18' if args.MoEx else cfg['model_name']
        cfg['StyleMix'] = args.StyleMix
        cfg['StyleMix_method'] = args.StyleMix_method
        assert args.StyleMix_method in ['StyleMix', 'StyleCutMix', 'StyleCutMix_Auto_Gamma']
        oss_proxy = OssProxy()
        if args.UMAx6:
            cfg['da_anno_file_list'] = []
            for editingName in ['M1L4', 'M1L8', 'M1L12', 'M2Ltop4', 'M2Ltop8', 'M2Ltop12']:
                json_path = 'leogb/causal/{}/editing{}.json'.format(cfg['train_file'].split('/')[-2], editingName)
                if oss_proxy.exists(json_path):
                    cfg['da_anno_file_list'].append(json_path)
        if args.UMAx10:
            cfg['da_anno_file_list'] = []
            for editingName in ['M1L4', 'M1L6', 'M1L8', 'M1L10', 'M1L12', 'M2Ltop4', 'M2Ltop6', 'M2Ltop8', 'M2Ltop10', 'M2Ltop12']:
            # for editingName in ['M2Ltop3', 'M2Ltop4', 'M2Ltop5', 'M2Ltop6', 'M2Ltop7', 'M2Ltop8', 'M2Ltop9', 'M2Ltop10', 'M2Ltop11', 'M2Ltop12']:
                json_path = 'leogb/causal/{}/editing{}.json'.format(cfg['train_file'].split('/')[-2], editingName)
                if oss_proxy.exists(json_path):
                    cfg['da_anno_file_list'].append(json_path)
        # if args.UMAx3:
        #     cfg['da_anno_file_list'] = []
        #     for editingName in ['M2Ltop4', 'M2Ltop8', 'M2Ltop12']:
        #         json_path = 'leogb/causal/{}/editing{}.json'.format(cfg['train_file'].split('/')[-2], editingName)
        #         if oss_proxy.exists(json_path):
        #             cfg['da_anno_file_list'].append(json_path)
        # if args.UMAx5:
        #     cfg['da_anno_file_list'] = []
        #     for editingName in ['M2Ltop4', 'M2Ltop6', 'M2Ltop8', 'M2Ltop10', 'M2Ltop12']:
        #         json_path = 'leogb/causal/{}/editing{}.json'.format(cfg['train_file'].split('/')[-2], editingName)
        #         if oss_proxy.exists(json_path):
        #             cfg['da_anno_file_list'].append(json_path)

        # if args.UMAx10_PLv1:
        #     cfg['da_anno_file_list'] = []
        #     for editingName in ['M1L4', 'M1L6', 'M1L8', 'M1L10', 'M1L12', 'M2Ltop4', 'M2Ltop6', 'M2Ltop8', 'M2Ltop10', 'M2Ltop12', 
        #         'M3L4_prelabel_095', 'M3L6_prelabel_095', 'M3L8_prelabel_095', 'M3L10_prelabel_095', 'M3L12_prelabel_095']:
        #         json_path = 'leogb/causal/{}/editing{}.json'.format(cfg['train_file'].split('/')[-2], editingName)
        #         if oss_proxy.exists(json_path):
        #             cfg['da_anno_file_list'].append(json_path)
        # if args.UMAx10_PLv2:
        #     cfg['da_anno_file_list'] = []
        #     for editingName in ['M1L4', 'M1L6', 'M1L8', 'M1L10', 'M1L12', 'M2Ltop4', 'M2Ltop6', 'M2Ltop8', 'M2Ltop10', 'M2Ltop12', 
        #         'M3L4_prelabel_095', 'M3L6_prelabel_095', 'M3L8_prelabel_095', 'M3L10_prelabel_095', 'M3L12_prelabel_095',
        #         'M4Ltop4_prelabel_095']:
        #         json_path = 'leogb/causal/{}/editing{}.json'.format(cfg['train_file'].split('/')[-2], editingName)
        #         if oss_proxy.exists(json_path):
        #             cfg['da_anno_file_list'].append(json_path)
        # if args.UMAx10_PLv3:
        #     cfg['da_anno_file_list'] = []
        #     for editingName in ['M1L4', 'M1L6', 'M1L8', 'M1L10', 'M1L12', 'M2Ltop4', 'M2Ltop6', 'M2Ltop8', 'M2Ltop10', 'M2Ltop12', 
        #         'M3L4_prelabel_095', 'M3L6_prelabel_095', 'M3L8_prelabel_095', 'M3L10_prelabel_095', 'M3L12_prelabel_095',
        #         'M4Ltop4_prelabel_095', 'M4Ltop6_prelabel_095', 'M4Ltop8_prelabel_095']:
        #         json_path = 'leogb/causal/{}/editing{}.json'.format(cfg['train_file'].split('/')[-2], editingName)
        #         if oss_proxy.exists(json_path):
        #             cfg['da_anno_file_list'].append(json_path)
        # if args.UMAx10_PLv4:
        #     cfg['da_anno_file_list'] = []
        #     for editingName in ['M1L4', 'M1L6', 'M1L8', 'M1L10', 'M1L12', 'M2Ltop4', 'M2Ltop6', 'M2Ltop8', 'M2Ltop10', 'M2Ltop12', 
        #         'M3L4_prelabel_095', 'M3L6_prelabel_095', 'M3L8_prelabel_095', 'M3L10_prelabel_095', 'M3L12_prelabel_095',
        #         'M4Ltop4_prelabel_095', 'M4Ltop6_prelabel_095', 'M4Ltop8_prelabel_095', 'M4Ltop10_prelabel_095', 'M4Ltop12_prelabel_095']:
        #         json_path = 'leogb/causal/{}/editing{}.json'.format(cfg['train_file'].split('/')[-2], editingName)
        #         if oss_proxy.exists(json_path):
        #             cfg['da_anno_file_list'].append(json_path)

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    if cfg['resume'] and cfg['pretrained_model']:
        print('='*50)
        print('Warning: cfg.resume:{} and cfg.pretrained_model:{} are not compatible. cfg.resume will be used training'.format(cfg['resume'], cfg['pretrained_model']))
        print('='*50)
    while psutil.pid_exists(args.wait_pid):
        print('%d is running, waiting...', args.wait_pid)
        time.sleep(100)
    return cfg

class GatherAndSelectFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, labels, logits):
        global_labels = [torch.zeros_like(labels) for _ in range(dist.get_world_size())]
        global_logits = [torch.zeros_like(logits) for _ in range(dist.get_world_size())]
        dist.all_gather(global_labels, labels)
        dist.all_gather(global_logits, logits)
        global_labels = torch.cat(global_labels, dim=0)
        global_logits = torch.cat(global_logits, dim=0)
        return global_labels, global_logits


class GatherAndSelectModule(nn.Module):
    def __init__(self):
        super(GatherAndSelectModule, self).__init__()

    def forward(self, labels, logits):
        return GatherAndSelectFunc.apply(labels, logits)


def build_loss(cfg):
    id_loss_func = F.cross_entropy
    # rank_loss_type = cfg.get('rank_loss_type', 'Triplet')
    # if rank_loss_type == 'Triplet':
        # triplet = TripletLoss(margin=cfg['margin'])
    # if rank_loss_type == 'MultiSimilarity':
        # triplet = MultiSimilarityLoss(margin=cfg['margin'])

    # def loss_func(logit, feat, target, hard_mining=True):
    #     id_loss = id_loss_func(logit, target)
    #     if cfg.get('sampler', 'RandomIdentitySampler') == 'RandomIdentitySampler':
    #         metric_loss = triplet(feat, target, cfg['normalize'], hard_mining)
    #         return id_loss * cfg['ce_weight'] + metric_loss * cfg['triplet_weight'], (id_loss, metric_loss)
    #     else:
    #         return id_loss * cfg['ce_weight'], (id_loss, 0.0) 

    return id_loss_func


def build_opm(cfg, model):
    if isinstance(model, list):
        params = list(model[0].parameters())
        for tt in model[1:]:
            params += list(tt.parameters())
    else:
        params = model.parameters()
    if cfg.get('freeze_bn', False):
        new_params = filter(lambda p: p.requires_grad, params)
    else:
        new_params = params
    optimizer = torch.optim.Adam(new_params, lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    if cfg['lr_scheduler'] == 'lr':
        scheduler = WarmupMultiStepLR(optimizer, milestones=cfg['milestones'], gamma=cfg['gamma'],warmup_factor=0.01, 
            warmup_iters=cfg['warmup_iters'], warmup_method="linear", last_epoch=-1)
    elif cfg['lr_scheduler'] == 'cos':
        scheduler = WarmupCosineLR(optimizer, max_epochs=['num_epochs'], warmup_epochs=cfg['warmup_iters'], eta_min=7e-7, last_epoch=-1)
    elif cfg['lr_scheduler'] == 'step':
        scheduler = StepLR(optimizer, step_size=cfg['step_size'], gamma=cfg['gamma'], last_epoch=-1)
    return optimizer, scheduler


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def train(cfg):
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

    # config log
    if 'rank' not in cfg:
        cfg['rank'] = 0
    logger = setup_logger(cfg['snap'], cfg['log_dir'], cfg['rank'])
    config_info(cfg, logger)

    # construct the dataset
    logger.info('=' * 10 + ' construct the dataset ' + '=' * 10)
    train_transform = build_transform(cfg, split='train')
    train_dataset = CLSDataset(
        cfg['train_file'], cfg, train_transform, split='train', delimiter=cfg['delimiter'], da_anno_file_list=cfg['da_anno_file_list'])
    cfg['id_nums'] = train_dataset.num_classes
    train_loader, train_sampler = get_dataloader(cfg, train_dataset, split='train')

    if cfg['valid']:
        valid_transform = build_transform(cfg, split='valid')
        valid_dataset = CLSDataset(
            cfg['valid_file'], cfg, valid_transform, split='valid', delimiter=cfg['delimiter'], da_anno_file_list=cfg['da_anno_file_list'])
        valid_loader, valid_sampler = get_dataloader(cfg, valid_dataset, split='valid')

    # constuct the network
    logger.info('=' * 10 + ' constuct the network ' + '=' * 10)
    img_encoder = ImgEncoder(cfg)
    if cfg['cuda']:
        img_encoder = img_encoder.cuda()

    # construct the optimizer & loss
    logger.info('=' * 10 + ' construct the optimizer & loss ' + '=' * 10)
    criterion = build_loss(cfg)
    optimizer, scheduler = build_opm(cfg, img_encoder)
    if cfg['resume']:
        logger.info("=> loading checkpoint '{}'".format(cfg['resume']))
        if os.path.isfile(cfg['resume']):
            checkpoint = torch.load(cfg['resume'])
        elif 'http' in cfg['resume']:
            checkpoint = model_zoo.load_url(cfg['resume'])
        else:
            oss_proxy = OssProxy()
            if not oss_proxy.exists(cfg['resume']):
                raise IOError('{} is not exists in oss'.format(cfg['resume']))
            else:
                with OssFile(cfg['resume']).get_bin_file() as f:
                    checkpoint = torch.load(f)
        start_epoch = checkpoint['epoch'] + 1
        if 'scheduler' in checkpoint.keys():
            logger.info('resume scheduler')
            scheduler.load_state_dict(checkpoint['scheduler'])
        img_encoder.load_state_dict(checkpoint['state_dict'])
        if 'optimizer' in checkpoint.keys():
            logger.info('resume optimizer')
            optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})" .format(cfg['resume'], checkpoint['epoch']))

    elif cfg['pretrained_model']:
        logger.info("=> loading pretrained model '{}'".format(cfg['pretrained_model']))
        if os.path.isfile(cfg['pretrained_model']):
            checkpoint = torch.load(cfg['pretrained_model'])
        elif 'http' in cfg['pretrained_model']:
            checkpoint = model_zoo.load_url(cfg['pretrained_model'])
        else:
            oss_proxy = OssProxy()
            if not oss_proxy.exists(cfg['pretrained_model']):
                raise IOError('{} is not exists in oss'.format(cfg['pretrained_model']))
            else:
                with OssFile(cfg['pretrained_model']).get_bin_file() as f:
                    checkpoint = torch.load(f)
            # checkpoint = model_zoo.load_url(cfg['pretrained_model'])
        start_epoch = 0
        if 'state_dict' in checkpoint.keys():
            img_encoder.load_param(checkpoint['state_dict'])
        else:
            img_encoder.load_param(checkpoint)

    else:
        start_epoch = 0

    # apex
    if cfg['apex']:
        print('using apex training')
        img_encoder, optimizer = amp.initialize(img_encoder, optimizer, opt_level="O1")

    # DataParallel Setup
    if not cfg.get('distributed', False):
        print('using DataParallel')
        img_encoder = nn.DataParallel(img_encoder)
        comm = None
    else:
        print('using DistributedDataParallel')
        if cfg.get('norm', 'BN') == 'SynBN':
            print('transforming BN to SynBN')
            img_encoder = nn.SyncBatchNorm.convert_sync_batchnorm(img_encoder)
        comm = GatherAndSelectModule()
        img_encoder = torch.nn.parallel.DistributedDataParallel(img_encoder, device_ids=[cfg['local_rank']])

    if cfg['valid']:
        logger.info('begin valid')
        c_right, c_all = validation(valid_loader, img_encoder, comm)
        logger.info('classification precision is {}/{} = {}%'.format(c_right, c_all, 100.0*c_right/c_all))

    # styleDistanceMatrix
    if cfg.get('StyleMix', False) and cfg.get('StyleMix_method', '')=='StyleCutMix_Auto_Gamma':
        dataset_name = cfg['train_file'].split('/')[-2]
        pt_path = 'leogb/causal_logs/{}/{}.pt'.format(dataset_name, dataset_name)
        print('load styleDistanceMatrix from {}'.format(pt_path))
        with OssFile(pt_path).get_bin_file() as f:
            styleDistanceMatrix = torch.load(f, map_location='cuda:0')
        styleDistanceMatrix = styleDistanceMatrix.cpu()
        ind = torch.arange(styleDistanceMatrix.shape[1])
        styleDistanceMatrix[ind, ind] += 2 # Prevent diagonal lines from zero

    if cfg.get('StyleMix', False) and cfg.get('StyleMix_method', '').startswith('Style'):
        import net_cutmix
        import net_mixup
        if cfg.get('StyleMix_method', '').startswith('StyleCutMix'):
            decoder = net_cutmix.decoder
            vgg = net_cutmix.vgg
            print("select network StyleCutMix")
            network_E = net_cutmix.Net_E(vgg)
            network_D = net_cutmix.Net_D(vgg, decoder)
        elif cfg.get('StyleMix_method', '') == 'StyleMix':
            decoder = net_mixup.decoder
            vgg = net_mixup.vgg
            print("select network StyleMix")
            network_E = net_mixup.Net_E(vgg)
            network_D = net_mixup.Net_D(vgg, decoder)
        else:
            raise Exception('unknown method: {}'.format(args.method))
        decoder.eval()
        vgg.eval()
        with OssFile('leogb/causal_code/models/decoder.pth.tar').get_bin_file() as f:
            decoder.load_state_dict(torch.load(f))
        with OssFile('leogb/causal_code/models/vgg_normalised.pth').get_bin_file() as f:
            vgg.load_state_dict(torch.load(f))
        vgg = nn.Sequential(*list(vgg.children())[:31])
        vgg.cuda()
        decoder.cuda()
        network_E.eval()
        network_D.eval()
        network_E = torch.nn.DataParallel(network_E).cuda()
        network_D = torch.nn.DataParallel(network_D).cuda()

    # start training
    img_counter = 0
    max_epochs = cfg['max_epochs']
    best_model_infos = {'best_accuracy': 0.0, 'best_model_path': ''}
    for epoch in range(start_epoch, max_epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        train_logger = LogCollector()
        logger.info('Epoch {}/{}'.format(epoch, max_epochs - 1))
        if cfg.get('empty_cache', False):
            torch.cuda.empty_cache()
        img_encoder.train()
        index = 0
        end = time.time()

        for name, inputs, labels in tqdm(train_loader):
            data_time.update(time.time() - end, 1)
            optimizer.zero_grad()
            if cfg['cuda']:
                inputs = inputs.cuda()
                labels = labels.cuda()

            if cfg.get('Mixup', False):
                logger.info('using Mixup ...')
                inputs, labels_a, labels_b, lam = mixup_data(inputs, labels)
                inputs, labels_a, labels_b = map(Variable, (inputs, labels_a, labels_b))

                # for imgidx in range(inputs.shape[0]):
                #     img = inputs[imgidx]
                #     img_save_path = './damap/mixup_{}_{}.jpg'.format(name[imgidx].split('/')[-1], img_counter)
                #     img_counter += 1
                #     result = tensor2im(img)
                #     Image.fromarray(np.array(result)).save(img_save_path)
                #     print('>>>>>>>>>>>>>>>> save damap to {}'.format(img_save_path))
                #     break

                logits = img_encoder(inputs)
                loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)

            elif cfg.get('Cutmix', False) and np.random.rand(1)<1.0:    #按照一定概率（cutmix_prob）进行cutmix
                logger.info('using Cutmix ...')
                inputs, labels_a, labels_b, lam = cutmix_data(inputs, labels)
                inputs, labels_a, labels_b = map(Variable, (inputs, labels_a, labels_b))

                # for imgidx in range(inputs.shape[0]):
                #     img = inputs[imgidx]
                #     img_save_path = './damap/cutmix_{}_{}.jpg'.format(name[imgidx].split('/')[-1], img_counter)
                #     img_counter += 1
                #     result = tensor2im(img)
                #     Image.fromarray(np.array(result)).save(img_save_path)
                #     print('>>>>>>>>>>>>>>>> save damap to {}'.format(img_save_path))
                #     break

                logits = img_encoder(inputs)
                loss = cutmix_criterion(criterion, logits, labels_a, labels_b, lam)

            elif cfg.get('MoEx', False):
                swap_index = torch.randperm(inputs.size(0), device=inputs.device)
                with torch.no_grad():
                    labels_a = labels
                    labels_b = labels[swap_index]

                # for imgidx in range(inputs.shape[0]):
                #     img = inputs[imgidx]
                #     img_save_path = './damap/moex_{}_{}.jpg'.format(name[imgidx].split('/')[-1], img_counter)
                #     img_counter += 1
                #     result = tensor2im(img)
                #     Image.fromarray(np.array(result)).save(img_save_path)
                #     print('>>>>>>>>>>>>>>>> save damap to {}'.format(img_save_path))
                #     break

                logits = img_encoder(inputs, swap_index=swap_index)
                lam = 0.9
                loss = criterion(logits, labels_a) * lam + criterion(logits, labels_b) * (1. - lam)

            elif cfg.get('Manifold_Mixup', False):
                logger.info('using Manifold_Mixup ...')
                inputs, labels = map(Variable, (inputs, labels))
                logits, reweighted_target = img_encoder(inputs, label=labels,is_train=True)
                bce_loss = nn.BCELoss().cuda()
                softmax = nn.Softmax(dim=1).cuda()
                loss = bce_loss(softmax(logits), reweighted_target)

            elif cfg.get('StyleMix', False):
                args_prob = 0
                args_delta = 3.0
                args_alpha2 = 1.0
                args_alpha1 = 1.0
                args_r = 0.7
                if 'StyleCutMix' in cfg.get('StyleMix_method', ''):
                    args_prob = 0.5
                else:
                    args_prob = 0.2
                prob = np.random.rand(1)
                if prob < args_prob:
                    rand_index = torch.randperm(inputs.size()[0]).cuda()
                    labels_1 = labels
                    labels_2 = labels[rand_index]
                    if cfg.get('StyleMix_method', '').startswith('StyleCutMix'):
                        if cfg.get('StyleMix_method', '') == 'StyleCutMix_Auto_Gamma':
                            styleDistance = styleDistanceMatrix[labels_1, labels_2]
                            gamma = torch.tanh(styleDistance/args_delta)
                        else :
                            args_alpha2 = 0.8
                            gamma = np.random.beta(args_alpha2, args_alpha2)

                        u = nn.Upsample(size=(224, 224), mode='bilinear')
                        x1 = u(inputs)
                        x2 = x1[rand_index]
                        rs = np.random.beta(args_alpha1, args_alpha1)
                        M = torch.zeros(1,1,224,224).float()
                        lam_temp = np.random.beta(args_alpha1, args_alpha1)
                        bbx1, bby1, bbx2, bby2 = rand_bbox(M.size(), 1.-lam_temp)
                        with torch.no_grad():
                            x1_feat = network_E(x1)
                            mixImage = network_D(x1, x2, x1_feat, x1_feat[rand_index], rs, gamma, bbx1, bby1, bbx2, bby2)
                        lam = ((bbx2 - bbx1)*(bby2-bby1)/(224.*224.))
                        # uinv = nn.Upsample(size=(32,32), mode='bilinear')
                        # logits = img_encoder(uinv(mixImage))

                        # for imgidx in range(mixImage.shape[0]):
                        #     img = mixImage[imgidx]
                        #     img_save_path = './damap/stylecutmix_{}_{}.jpg'.format(name[imgidx].split('/')[-1], img_counter)
                        #     img_counter += 1
                        #     result = tensor2im(img)
                        #     Image.fromarray(np.array(result)).save(img_save_path)
                        #     print('>>>>>>>>>>>>>>>> save damap to {}'.format(img_save_path))
                        #     break

                        logits = img_encoder(mixImage)

                        log_preds = F.log_softmax(logits, dim=-1) # dimension [batch_size, numberofclass]
                        a_loss = -log_preds[torch.arange(logits.shape[0]), labels_1] # cross-entropy for A
                        b_loss = -log_preds[torch.arange(logits.shape[0]), labels_2] # cross-entropy for B
                        if cfg.get('StyleMix_method', '') == 'StyleCutMix_Auto_Gamma':
                            gamma = gamma.cuda()
                        lam_s = gamma * lam + (1.0 - gamma) * rs
                        loss_c = a_loss * (lam) + b_loss * (1. - lam)
                        loss_s = a_loss * (lam_s) + b_loss * (1. - lam_s)
                        loss = (args_r * loss_c + (1.0 - args_r) * loss_s).mean()

                    elif cfg.get('StyleMix_method', '') == 'StyleMix':
                        args_alpha1 = 0.5

                        u = nn.Upsample(size=(224, 224), mode='bilinear')
                        x1 = u(inputs)
                        x2 = x1[rand_index]
                        rc = np.random.beta(args_alpha1, args_alpha1)
                        rs = np.random.beta(args_alpha1, args_alpha1)
                        with torch.no_grad():
                            x1_feat = network_E(x1)
                            mixImage = network_D(x1_feat, x1_feat[rand_index], rc, rs)
                        # uinv = nn.Upsample(size=(32,32), mode='bilinear')
                        # logits = img_encoder(uinv(mixImage))

                        # for imgidx in range(mixImage.shape[0]):
                        #     img = mixImage[imgidx]
                        #     img_save_path = './damap/stylemix_{}_{}.jpg'.format(name[imgidx].split('/')[-1], img_counter)
                        #     img_counter += 1
                        #     result = tensor2im(img)
                        #     Image.fromarray(np.array(result)).save(img_save_path)
                        #     print('>>>>>>>>>>>>>>>> save damap to {}'.format(img_save_path))
                        #     break

                        logits = img_encoder(mixImage)

                        loss_c = rc * criterion(logits, labels_1)  + (1.0 - rc) * criterion(logits, labels_2)
                        loss_s = rs * criterion(logits, labels_1)  + (1.0 - rs) * criterion(logits, labels_2)
                        loss = args_r * loss_c + (1.0-args_r) * loss_s
                else:
                    logits = img_encoder(inputs)
                    loss = criterion(logits, labels)

            else:
                logits = img_encoder(inputs)
                # pooled_feats->Bxfeat_dimsxwxh, logits->Bxfeat_dims
                loss = criterion(logits, labels)

            # apex
            if cfg.get('apex', True):
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()

            train_logger.update('loss', loss.item(), logits.size(0))
            batch_time.update(time.time() - end, 1)
            end = time.time()
            if index % cfg['print_interval'] == 0:
                logger.info('Epoch: [{0}][{1}/{2}]\tlr{3}\t'
                            '{e_log}\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                             .format(
                                    epoch, index, len(train_loader), scheduler.get_lr()[0],
                                    batch_time=batch_time, data_time=data_time, e_log=str(train_logger)))
            index += 1
        if scheduler is not None:  # after val at this epoch
            scheduler.step()

        if epoch % cfg['valid_interval'] == 0 and cfg['valid']:
            logger.info('begin valid')
            if cfg.get('empty_cache', False):
                torch.cuda.empty_cache()
            c_right, c_all = validation(valid_loader, img_encoder, comm)
            logger.info('classification precision is {}/{} = {}%'.format(c_right, c_all, 100.0*c_right/c_all))

        if cfg['rank'] == 0:
            upload_start_time = time.time()
            save_path = os.path.join(cfg['log_dir'], cfg['snap'],'ckpt_ep{}.pth'.format(epoch))
            ckpt = {
                'epoch': epoch,
                'state_dict': img_encoder.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            ckp_buf = BytesIO()
            torch.save(ckpt, ckp_buf)
            oss_proxy = OssProxy()
            oss_proxy.upload_str_to_oss(ckp_buf.getvalue(), save_path)
            logger.info('Save checkpoints to {} with {} s'.format(save_path, time.time() - upload_start_time))

            if 100.0*c_right/c_all > best_model_infos['best_accuracy']:
                best_model_infos['best_accuracy'] = 100.0*c_right/c_all
                best_model_infos['best_model_path'] = save_path
            save_json(os.path.join(cfg['log_dir'], cfg['snap'], 'best_model_infos.json'), best_model_infos)


def validation(valid_loader, img_encoder, comm=None):
    img_encoder.eval()
    classify_correct = 0
    classify_total = 0
    for i, inputs in tqdm(enumerate(valid_loader), total=len(valid_loader)):
        name, img, labels = inputs
        img = img.cuda()
        labels = labels.cuda()
        with torch.no_grad():
            logit = img_encoder(img)
        global_labels, global_logits, = comm(labels, logit)
        npy_labels = global_labels.cpu().numpy()
        npy_logit = global_logits.cpu().numpy()
        predicted = np.argmax(npy_logit, 1)
        classify_correct += (predicted == npy_labels).sum()
        classify_total += len(npy_labels)
    img_encoder.train()

    return classify_correct, classify_total


def main(cfg):
    if cfg.get('distributed', False):
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method('spawn')
        rank = cfg['local_rank']
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(rank % num_gpus)
        dist.init_process_group(backend='nccl')
        cfg['world_size'] = get_world_size()
        cfg['rank'] = get_rank()
        print('local_rank of this process is {}, rank {}, world_size {}'.format(rank, cfg['rank'], cfg['world_size']))

    if cfg['mode'] == 'train':
        print('start train')
        train(cfg)


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
