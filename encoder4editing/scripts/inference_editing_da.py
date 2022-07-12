import argparse

import torch
import numpy as np
import os, sys, json, pickle, time
import dlib

sys.path.append(".")
sys.path.append("..")

from configs import data_configs, paths_config
from datasets.inference_dataset import InferenceDataset, InferenceDataset_causal
from torch.utils.data import DataLoader
from utils.model_utils import setup_model
from utils.common import tensor2im
from utils.alignment import align_face
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
from oss import OssProxy, OssFile
from io import BytesIO
from tqdm import tqdm


def main(args):
    net, opts = setup_model(args.ckpt, device)
    is_cars = 'cars_' in opts.dataset_type
    generator = net.decoder
    generator.eval()
    args, data_loader = setup_data_loader(args, opts)

    src_directory_path = os.path.join(args.save_dir)
    os.makedirs(src_directory_path, exist_ok=True)

    # Check if latents exist
    latents_file_path = os.path.join(args.save_dir, 'latents.pt')
    if os.path.exists(latents_file_path):
        latent_codes = torch.load(latents_file_path).to(device)
        with open(os.path.join(args.save_dir, 'alldatainfo.pickle'), 'rb') as f:
            all_paths, all_targets = pickle.load(f)
    else:
        latent_codes, all_paths, all_targets = get_all_latents(net, data_loader, args.n_sample, is_cars=is_cars)
        # torch.save(latent_codes, latents_file_path)
        # with open(os.path.join(args.save_dir, 'alldatainfo.pickle'), 'wb') as f:
        #     pickle.dump([all_paths, all_targets], f)

    print('latent_codes.shape: {}'.format(latent_codes.shape))
    print('len(all_paths): {}'.format(len(all_paths)))
    print('len(all_targets): {}'.format(len(all_targets)))

    if not args.latents_only:
        generate_inversions_causal_da_editing1(args, generator, latent_codes, all_paths, all_targets, is_cars=is_cars)
        # generate_inversions_causal_da_editing2(args, generator, latent_codes, all_paths, all_targets, is_cars=is_cars)

    # src_directory_path = os.path.join(args.save_dir, 'src')
    # os.makedirs(src_directory_path, exist_ok=True)


def setup_data_loader(args, opts):
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    anno_file = args.anno_file
    print(f"anno_file: {anno_file}")
    align_function = None
    if args.align:
        align_function = run_alignment
    test_dataset = InferenceDataset_causal(anno_file=anno_file,
                                    transform=transforms_dict['transform_test'],
                                    preprocess=align_function,
                                    opts=opts)
    data_loader = DataLoader(test_dataset,
                             batch_size=args.batch,
                             shuffle=False,
                             num_workers=32,
                             drop_last=True)
    print(f'dataset length: {len(test_dataset)}')

    if args.n_sample is None:
        args.n_sample = len(test_dataset)

    return args, data_loader


def run_alignment(image_path):
    predictor = dlib.shape_predictor(paths_config.model_paths['shape_predictor'])
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image


# def image_preprocess(opts, from_path):
#     dataset_args = data_configs.DATASETS[opts.dataset_type]
#     transforms_dict = dataset_args['transforms'](opts).get_transforms()
#     transform = transforms_dict['transform_test']
#     from_im = Image.open(from_path).convert('RGB')
#     from_im = transform(from_im)
#     return from_im


def get_latents(net, x, is_cars=False):
    codes = net.encoder(x)
    if net.opts.start_from_latent_avg:
        if codes.ndim == 2:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
        else:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)
    if codes.shape[1] == 18 and is_cars:
        codes = codes[:, :16, :]
    return codes


def get_all_latents(net, data_loader, n_images=None, is_cars=False):
    all_latents = []
    all_paths = []
    all_targets = []
    i = 0
    with torch.no_grad():
        for batch_index, (batch_path, batch_img, batch_target) in tqdm(enumerate(data_loader), total=len(data_loader)):
            if n_images is not None and i > n_images:
                break
            x = batch_img
            inputs = x.to(device).float()
            latents = get_latents(net, inputs, is_cars)
            all_latents.append(latents)
            all_paths += batch_path
            all_targets += batch_target
            i += len(latents)
    return torch.cat(all_latents), all_paths, all_targets


def save_image(img, save_dir, idx):
    result = tensor2im(img)
    im_save_path = os.path.join(save_dir, f"{idx:05d}.jpg")
    Image.fromarray(np.array(result)).save(im_save_path)


def save_image_withname(img, save_dir, imgname):
    result = tensor2im(img)
    im_save_path = os.path.join(save_dir, imgname)
    Image.fromarray(np.array(result)).save(im_save_path)


@torch.no_grad()
def generate_inversions(args, g, latent_codes, is_cars):
    print('Saving inversion images')
    inversions_directory_path = os.path.join(args.save_dir, 'inversions')
    os.makedirs(inversions_directory_path, exist_ok=True)
    for i in range(args.n_sample):
        imgs, _ = g([latent_codes[i].unsqueeze(0)], input_is_latent=True, randomize_noise=False, return_latents=True)
        if is_cars:
            imgs = imgs[:, :, 64:448, :]
        save_image(imgs[0], inversions_directory_path, i + 1)


def gen_splice_map(save_dir, imgname, imglistdir, save_path, img_w=512, img_h=384, boardersize=40):
    img_src = Image.open(os.path.join(save_dir, 'src', imgname)).convert('RGB')
    img_src = img_src.resize((img_w, img_h))
    img_inv = Image.open(os.path.join(save_dir, 'inversions', imgname)).convert('RGB')

    imglist = os.listdir(os.path.join(save_dir, imglistdir))
    imglist = [i for i in imglist if imgname.split('.')[0] in i]
    imglist = sorted(imglist)

    target = Image.new('RGB', (img_w*(len(imglist)+2), img_h+boardersize))
    draw = ImageDraw.Draw(target)
    this_dir = os.path.abspath(os.path.dirname(__file__))
    font = ImageFont.truetype(os.path.join(this_dir, 'simsun.ttc'), boardersize)
    draw.text((img_w*0, 0), "src", (255,255,255), font=font)
    target.paste(img_src, (img_w*0, boardersize))
    draw.text((img_w*1, 0), "inv", (255,255,255), font=font)
    target.paste(img_inv, (img_w*1, boardersize))
    for idx, nm in enumerate(imglist):
        draw.text((img_w*(2+idx), 0), "L{}".format(idx), (255,255,255), font=font)
        img_edt = Image.open(os.path.join(save_dir, imglistdir, nm)).convert('RGB')
        target.paste(img_edt, (img_w*(2+idx), boardersize))

    target.save(save_path)


@torch.no_grad()
def generate_inversions_causal_da_editing1(args, g, latent_codes, all_paths, all_targets, is_cars): #args, g, index_pair, latent_codes, img_paths, is_cars
    
    # 'leogb/causal/CelebAMask-HQ/CelebAMask-HQ-9_editingM1L8.json',      #同类，单层交换
    # 'leogb/causal/CelebAMask-HQ/CelebAMask-HQ-9_editingM2Ltop8.json',   #同类，多层交换
    # 'leogb/causal/CelebAMask-HQ/CelebAMask-HQ-9_editingM3L4.json',      #不同类，单层交换
    # 'leogb/causal/CelebAMask-HQ/CelebAMask-HQ-9_editingM4Ltop4.json',   #不同类，多层交换

    save_folder = args.save_dir.split('/')[-2]


    # ============================================================
    # 1. 同类，单层交换
    # ============================================================
    for layer in range(0, 14, 1): #[4,6,8,10,12]:
        editing_name = 'editingM1L{}'.format(layer)
        print('generating for {}...'.format(editing_name))
        editing_img_dict = {}
        src_directory_path = os.path.join(args.save_dir, editing_name)
        os.makedirs(src_directory_path, exist_ok=True)

        for bs in tqdm(range(len(all_paths)//args.batch)):
            cur_all_paths = all_paths[bs*args.batch : (bs+1)*args.batch]
            cur_all_targets = all_targets[bs*args.batch : (bs+1)*args.batch]
            cur_latent_codes = latent_codes[bs*args.batch : (bs+1)*args.batch].clone()
            print(bs, len(cur_all_paths), len(cur_all_targets), cur_latent_codes.shape)
            for i in range(len(cur_all_paths)):
                path_1 = cur_all_paths[i]
                target_1 = cur_all_targets[i].item()
                latcode_1 = cur_latent_codes[i].clone()
                path_2 = cur_all_paths[(i+2)%args.batch]
                target_2 = cur_all_targets[(i+2)%args.batch].item()
                latcode_2 = cur_latent_codes[(i+2)%args.batch].clone()
                assert target_1==target_2   #同类

                latcode_1[layer] = latcode_2[layer]   #单层
                imgs, _ = g([latcode_1.unsqueeze(0)], input_is_latent=True, randomize_noise=False, return_latents=True)
                if is_cars:
                    imgs = imgs[:, :, 64:448, :]

                key = path_1.split('/')[-1]
                save_img_name = '{}_{}_{}.jpg'.format(path_1.split('/')[-1].split('.')[0], path_2.split('/')[-1].split('.')[0], target_1)
                save_image_withname(imgs[0], src_directory_path, save_img_name)
                editing_img_dict[key] = ['leogb/causal/{}/{}/'.format(save_folder, editing_name)+save_img_name, target_1]

        with open(os.path.join(args.save_dir, '{}.json'.format(editing_name)), 'w') as f:
            json.dump(editing_img_dict, f)

    # ============================================================
    # 2. 同类，多层交换
    # ============================================================
    for layer in range(0, 14, 1): #[4,6,8,10,12]:
        editing_name = 'editingM2Ltop{}'.format(layer)
        print('generating for {}...'.format(editing_name))
        editing_img_dict = {}
        src_directory_path = os.path.join(args.save_dir, editing_name)
        os.makedirs(src_directory_path, exist_ok=True)

        for bs in tqdm(range(len(all_paths)//args.batch)):
            cur_all_paths = all_paths[bs*args.batch : (bs+1)*args.batch]
            cur_all_targets = all_targets[bs*args.batch : (bs+1)*args.batch]
            cur_latent_codes = latent_codes[bs*args.batch : (bs+1)*args.batch].clone()
            print(bs, len(cur_all_paths), len(cur_all_targets), cur_latent_codes.shape)
            for i in range(len(cur_all_paths)):
                path_1 = cur_all_paths[i]
                target_1 = cur_all_targets[i].item()
                latcode_1 = cur_latent_codes[i].clone()
                path_2 = cur_all_paths[(i+2)%args.batch]
                target_2 = cur_all_targets[(i+2)%args.batch].item()
                latcode_2 = cur_latent_codes[(i+2)%args.batch].clone()
                assert target_1==target_2   #同类

                for l in range(layer+1):
                    latcode_1[l] = latcode_2[l]   #多层
                imgs, _ = g([latcode_1.unsqueeze(0)], input_is_latent=True, randomize_noise=False, return_latents=True)
                if is_cars:
                    imgs = imgs[:, :, 64:448, :]

                key = path_1.split('/')[-1]
                save_img_name = '{}_{}_{}.jpg'.format(path_1.split('/')[-1].split('.')[0], path_2.split('/')[-1].split('.')[0], target_1)
                save_image_withname(imgs[0], src_directory_path, save_img_name)
                editing_img_dict[key] = ['leogb/causal/{}/{}/'.format(save_folder, editing_name)+save_img_name, target_1]

        with open(os.path.join(args.save_dir, '{}.json'.format(editing_name)), 'w') as f:
            json.dump(editing_img_dict, f)

    # ============================================================
    # 3. 不同类，单层交换
    # ============================================================
    for layer in range(0, 18, 2): #[4,6,8,10,12]:
        editing_name = 'editingM3L{}'.format(layer)
        print('generating for {}...'.format(editing_name))
        editing_img_dict = {}
        src_directory_path = os.path.join(args.save_dir, editing_name)
        os.makedirs(src_directory_path, exist_ok=True)

        for bs in tqdm(range(len(all_paths)//args.batch+1)):
            cur_all_paths = all_paths[bs*args.batch : (bs+1)*args.batch]
            cur_all_targets = all_targets[bs*args.batch : (bs+1)*args.batch]
            cur_latent_codes = latent_codes[bs*args.batch : (bs+1)*args.batch].clone()
            print(bs, len(cur_all_paths), len(cur_all_targets), cur_latent_codes.shape)
            for i in range(len(cur_all_paths)):
                path_1 = cur_all_paths[i]
                target_1 = cur_all_targets[i].item()
                latcode_1 = cur_latent_codes[i].clone()
                path_2 = cur_all_paths[(i+1)%args.batch]
                target_2 = cur_all_targets[(i+1)%args.batch].item()
                latcode_2 = cur_latent_codes[(i+1)%args.batch].clone()
                assert target_1!=target_2   #不同类

                latcode_1[layer] = latcode_2[layer]   #单层
                imgs, _ = g([latcode_1.unsqueeze(0)], input_is_latent=True, randomize_noise=False, return_latents=True)
                if is_cars:
                    imgs = imgs[:, :, 64:448, :]

                key = path_1.split('/')[-1]
                save_img_name = '{}_{}_{}.jpg'.format(path_1.split('/')[-1].split('.')[0], path_2.split('/')[-1].split('.')[0], target_1)
                save_image_withname(imgs[0], src_directory_path, save_img_name)
                editing_img_dict[key] = ['leogb/causal/{}/{}/'.format(save_folder, editing_name)+save_img_name, target_1]

        with open(os.path.join(args.save_dir, '{}.json'.format(editing_name)), 'w') as f:
            json.dump(editing_img_dict, f)

    # ============================================================
    # 4. 不同类，多层交换
    # ============================================================
    for layer in range(0, 18, 2): #[4,6,8,10,12]:
        editing_name = 'editingM4Ltop{}'.format(layer)
        print('generating for {}...'.format(editing_name))
        editing_img_dict = {}
        src_directory_path = os.path.join(args.save_dir, editing_name)
        os.makedirs(src_directory_path, exist_ok=True)

        for bs in tqdm(range(len(all_paths)//args.batch+1)):
            cur_all_paths = all_paths[bs*args.batch : (bs+1)*args.batch]
            cur_all_targets = all_targets[bs*args.batch : (bs+1)*args.batch]
            cur_latent_codes = latent_codes[bs*args.batch : (bs+1)*args.batch].clone()
            print(bs, len(cur_all_paths), len(cur_all_targets), cur_latent_codes.shape)
            for i in range(len(cur_all_paths)):
                path_1 = cur_all_paths[i]
                target_1 = cur_all_targets[i].item()
                latcode_1 = cur_latent_codes[i].clone()
                path_2 = cur_all_paths[(i+1)%args.batch]
                target_2 = cur_all_targets[(i+1)%args.batch].item()
                latcode_2 = cur_latent_codes[(i+1)%args.batch].clone()
                assert target_1!=target_2   #不同类

                for l in range(layer+1):
                    latcode_1[l] = latcode_2[l]   #多层
                imgs, _ = g([latcode_1.unsqueeze(0)], input_is_latent=True, randomize_noise=False, return_latents=True)
                if is_cars:
                    imgs = imgs[:, :, 64:448, :]

                key = path_1.split('/')[-1]
                save_img_name = '{}_{}_{}.jpg'.format(path_1.split('/')[-1].split('.')[0], path_2.split('/')[-1].split('.')[0], target_1)
                save_image_withname(imgs[0], src_directory_path, save_img_name)
                editing_img_dict[key] = ['leogb/causal/{}/{}/'.format(save_folder, editing_name)+save_img_name, target_1]

        with open(os.path.join(args.save_dir, '{}.json'.format(editing_name)), 'w') as f:
            json.dump(editing_img_dict, f)


@torch.no_grad()
def generate_inversions_causal_da_editing2(args, g, latent_codes, all_paths, all_targets, is_cars): #args, g, index_pair, latent_codes, img_paths, is_cars
    
    # 'leogb/causal/CelebAMask-HQ/CelebAMask-HQ-9_editingM1L8.json',      #同类，单层交换
    # 'leogb/causal/CelebAMask-HQ/CelebAMask-HQ-9_editingM2Ltop8.json',   #同类，多层交换
    # 'leogb/causal/CelebAMask-HQ/CelebAMask-HQ-9_editingM3L4.json',      #不同类，单层交换
    # 'leogb/causal/CelebAMask-HQ/CelebAMask-HQ-9_editingM4Ltop4.json',   #不同类，多层交换
    # 'leogb/causal/CelebAMask-HQ/CelebAMask-HQ-9_editingM5.json',        #全部组合

    save_folder = args.save_dir.split('/')[-2]

    editing_name = 'editingM5'
    print('generating for {}...'.format(editing_name))
    editing_img_dict = {}
    src_directory_path = os.path.join(args.save_dir, editing_name)
    os.makedirs(src_directory_path, exist_ok=True)

    for bs in range(len(all_paths)//args.batch):
        cur_all_paths = all_paths[bs*args.batch : (bs+1)*args.batch]
        cur_all_targets = all_targets[bs*args.batch : (bs+1)*args.batch]
        cur_latent_codes = latent_codes[bs*args.batch : (bs+1)*args.batch].clone()
        print(bs, len(cur_all_paths), len(cur_all_targets), cur_latent_codes.shape)
        for i in tqdm(range(len(cur_all_paths))):
            path_1 = cur_all_paths[i]
            target_1 = cur_all_targets[i].item()
            latcode_1 = cur_latent_codes[i].clone()

            for j in range(len(cur_all_paths)):
                path_2 = cur_all_paths[j]
                target_2 = cur_all_targets[j].item()
                latcode_2 = cur_latent_codes[j].clone()

                if target_1!=target_2:
                    continue

                for layer in [8]:#[4,6,8,10,12]:
                    tmp_latcode_1 = latcode_1.clone()
                    tmp_latcode_1[layer] = latcode_2[layer]   #单层
                    # st = time.time()
                    imgs, _ = g([tmp_latcode_1.unsqueeze(0)], input_is_latent=True, randomize_noise=False, return_latents=True)
                    # print('g-s time:', time.time()-st)
                    if is_cars:
                        imgs = imgs[:, :, 64:448, :]
                    key = path_1.split('/')[-1]
                    save_img_name = '{}_{}_S{}_{}.jpg'.format(path_1.split('/')[-1].split('.')[0], path_2.split('/')[-1].split('.')[0], layer, target_1)
                    # st = time.time()
                    save_image_withname(imgs[0], src_directory_path, save_img_name)
                    # print('save time:', time.time()-st)
                    editing_img_dict.setdefault(key, [])
                    editing_img_dict[key].append(['leogb/causal/{}/{}/'.format(save_folder, editing_name)+save_img_name, target_1])

                for layer in [4,8,12]:#[4,6,8,10,12]:
                    tmp_latcode_1 = latcode_1.clone()
                    for l in range(layer+1):
                        tmp_latcode_1[l] = latcode_2[l]   #多层
                    # st = time.time()
                    imgs, _ = g([tmp_latcode_1.unsqueeze(0)], input_is_latent=True, randomize_noise=False, return_latents=True)
                    # print('g-m time:', time.time()-st)
                    if is_cars:
                        imgs = imgs[:, :, 64:448, :]
                    key = path_1.split('/')[-1]
                    save_img_name = '{}_{}_M{}_{}.jpg'.format(path_1.split('/')[-1].split('.')[0], path_2.split('/')[-1].split('.')[0], layer, target_1)
                    # st = time.time()
                    save_image_withname(imgs[0], src_directory_path, save_img_name)
                    # print('save time:', time.time()-st)
                    editing_img_dict.setdefault(key, [])
                    editing_img_dict[key].append(['leogb/causal/{}/{}/'.format(save_folder, editing_name)+save_img_name, target_1])

                # # st = time.time()
                # batch_latcode_1 = []
                # save_img_name_list = []
                # for layer in [4,6,8,10,12]:
                #     tmp_latcode_1 = latcode_1.clone()
                #     tmp_latcode_1[layer] = latcode_2[layer]   #单层
                #     batch_latcode_1.append(tmp_latcode_1)
                #     save_img_name = '{}_{}_S{}_{}.jpg'.format(path_1.split('/')[-1].split('.')[0], path_2.split('/')[-1].split('.')[0], layer, target_1)
                #     save_img_name_list.append(save_img_name)
                # for layer in [4,6,8,10,12]:
                #     tmp_latcode_1 = latcode_1.clone()
                #     for l in range(layer+1):
                #         tmp_latcode_1[l] = latcode_2[l]   #多层
                #     batch_latcode_1.append(tmp_latcode_1)
                #     save_img_name = '{}_{}_M{}_{}.jpg'.format(path_1.split('/')[-1].split('.')[0], path_2.split('/')[-1].split('.')[0], layer, target_1)
                #     save_img_name_list.append(save_img_name)

                # batch_latcode_1 = torch.stack(batch_latcode_1)
                # imgs, _ = g([batch_latcode_1], input_is_latent=True, randomize_noise=False, return_latents=True)
                # if is_cars:
                #     imgs = imgs[:, :, 64:448, :]
                # # print('g-batch time:', time.time()-st)

                # # st = time.time()
                # key = path_1.split('/')[-1]
                # editing_img_dict.setdefault(key, [])
                # for imgindex in range(len(save_img_name_list)):
                #     save_img_name = save_img_name_list[imgindex]
                #     save_image_withname(imgs[imgindex], src_directory_path, save_img_name)
                #     editing_img_dict[key].append(['leogb/causal/{}/{}/'.format(save_folder, editing_name)+save_img_name, target_1])
                # # print('save-batch time:', time.time()-st)

    with open(os.path.join(args.save_dir, '{}.json'.format(editing_name)), 'w') as f:
        json.dump(editing_img_dict, f)


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--anno_file", type=str, default='leogb/causal_logs/CelebAMask-HQ-9_v2/train_attrbute_9.txt', help="The directory to save the latent codes and inversion images. (default: images_dir")
    parser.add_argument("--save_dir", type=str, default='./testdata_output/train_attrbute_9/', help="")
    parser.add_argument("--batch", type=int, default=32, help="batch size for the generator")
    parser.add_argument("--n_sample", type=int, default=None, help="number of the samples to infer.")
    parser.add_argument("--latents_only", action="store_true", help="infer only the latent codes of the directory")
    parser.add_argument("--align", action="store_true", help="align face images before inference")
    parser.add_argument("ckpt", metavar="CHECKPOINT", help="path to generator checkpoint")
    args = parser.parse_args()
    main(args)
