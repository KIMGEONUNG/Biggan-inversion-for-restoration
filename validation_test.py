#!/usr/bin/env python3

import os
from skimage import color
import torch
from torchvision.transforms import ToPILImage, ToTensor
import torchvision.transforms as transforms
from pytorch_pretrained_biggan import (one_hot_from_int,
        truncated_noise_sample)
from tqdm import tqdm

import argparse
from torchvision.utils import make_grid

import numpy as np

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from utils import set_seed
from train_encoder_f_imgnet import Inversion


DEV = 'cuda'


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--num_row', type=int, default=16)
    parser.add_argument('--limit_iter', type=int, default=100)
    parser.add_argument('--size_batch', type=int, default=16)
    parser.add_argument('--truncation', type=float, default=1.0)

    parser.add_argument('--use_train_data', action='store_true')

    # I/O
    parser.add_argument('--path_dataset_val', type=str,
            default='dataset_val')
    parser.add_argument('--path_output_val', type=str,
            default='validation_data_result')

    parser.add_argument('--path_dataset_train', type=str,
            default='dataset_train')
    parser.add_argument('--path_output_train', type=str,
            default='train_data_result')

    parser.add_argument('--path_ckpt_encoder', type=str,
            default='./encoder_f.ckpt')

    return parser.parse_args()


def fusion(x_l, x_ab):
    labs = []
    for img_gt, img_hat in zip(x_l, x_ab):

        img_gt = img_gt.permute(1, 2, 0)
        img_hat = img_hat.permute(1, 2, 0)

        img_gt = color.rgb2lab(img_gt)
        img_hat = color.rgb2lab(img_hat)

        l = img_gt[:, :, :1]
        ab = img_hat[:, :, 1:]
        img_fusion = np.concatenate((l, ab), axis=-1)
        img_fusion = color.lab2rgb(img_fusion)
        img_fusion = torch.from_numpy(img_fusion)
        img_fusion = img_fusion.permute(2, 0, 1)
        labs.append(img_fusion)
    labs = torch.stack(labs)

    return labs


def main(args):
    print(args)

    if args.seed >= 0:
        set_seed(args.seed)

    # Model
    model = Inversion()
    model.encoder.load_state_dict(
            torch.load(args.path_ckpt_encoder))
    model.eval()
    model.to(DEV)

    # Datasets
    prep = transforms.Compose([
                ToTensor(),
                transforms.Resize(256),
                transforms.CenterCrop(256),
            ])

    if args.use_train_data:
        path_output = args.path_output_train
        path_dataset = args.path_dataset_train
        is_shuffle = True
    else:
        path_output = args.path_output_val
        path_dataset = args.path_dataset_val
        is_shuffle = True

    dataset = ImageFolder(path_dataset, transform=prep)
    dataloader = DataLoader(dataset, batch_size=args.size_batch,
            shuffle=is_shuffle, num_workers=8, drop_last=True)

    if not os.path.exists(path_output):
        os.mkdir(path_output)

    print('Iteration limit:', args.limit_iter)

    with torch.no_grad():
        for i, (x, c) in enumerate(tqdm(dataloader)):

            # Preserve GT
            x_gt = x.to(DEV)
            grid_gt = make_grid(x_gt, nrow=args.num_row)

            # Input
            x = x.to(DEV)
            x = transforms.Grayscale()(x)
            grid_gray = make_grid(x, nrow=args.num_row)

            # Latents
            c = one_hot_from_int(c, batch_size=args.size_batch)
            c = torch.from_numpy(c)
            c = c.to(DEV)

            # Noise
            z = truncated_noise_sample(truncation=args.truncation,
                    batch_size=args.size_batch)
            z = torch.from_numpy(z)
            z = z.to(DEV)

            # Inference
            output = model(x, z, c)
            grid_out = make_grid(output, nrow=args.num_row)

            # LAB
            labs = fusion(x_gt.detach().cpu(), output.detach().cpu())
            grid_lab = make_grid(labs, nrow=args.num_row).to(DEV)

            grid = torch.cat([grid_gt, grid_gray, grid_out, grid_lab], dim=-2)
            im = ToPILImage()(grid)

            path_result = os.path.join(path_output, '%03d.jpg' % (i))
            im.save(path_result)
            if i >= args.limit_iter:
                break


if __name__ == '__main__':
    args = parse()
    main(args)
