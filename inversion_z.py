#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor
import torchvision.transforms as transforms
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_int,
        truncated_noise_sample)
from tqdm import tqdm
import matplotlib.pyplot as plt

import os
import os.path
from os.path import join
import argparse

from model import VGG16Perceptual
import numpy as np
import random

DEV = 'cuda'


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=1)
    # 5 --> 32, 4 --> 16, ...
    parser.add_argument('--num_feat_layer', type=int, default=4)
    parser.add_argument('--resolution', type=str, default='256')
    parser.add_argument('--class_index', type=int, default=15)
    parser.add_argument('--iter', type=int, default=10000)
    parser.add_argument('--interval_save', type=int, default=10)
    parser.add_argument('--size_batch', type=int, default=1)
    parser.add_argument('--truncation', type=float, default=1.0)
    parser.add_argument('--show', action='store_false')

    # I/O
    parser.add_argument('--path_history', type=str, default='inversion_zf_hist')
    parser.add_argument('--path_target', type=str, default='./real.jpg')

    # Loss
    parser.add_argument('--loss_mse', action='store_true', default=True)
    parser.add_argument('--loss_lpips', action='store_true', default=False)
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def main(args):
    print(args)

    if not os.path.exists(args.path_history):
        os.mkdir(args.path_history)

    if args.seed >= 0:
        set_seed(args.seed)

    # Logger
    writer = SummaryWriter('runs/gray_inversion_z')

    im = Image.open(args.path_target)
    target = ToTensor()(im).unsqueeze(0).to(DEV)
    writer.add_image('GT', target.squeeze(0))

    class_vector = one_hot_from_int([args.class_index],
            batch_size=args.size_batch)

    noise_vector = truncated_noise_sample(truncation=args.truncation,
            batch_size=args.size_batch)

    noise_vector = torch.from_numpy(noise_vector)
    class_vector = torch.from_numpy(class_vector)

    # If you have a GPU, put everything on cuda
    noise_vector = noise_vector.to(DEV)
    class_vector = class_vector.to(DEV)

    noise_vector.requires_grad = True

    # Model
    name_model = 'biggan-deep-%s' % (args.resolution)
    model = BigGAN.from_pretrained(name_model)
    model.to(DEV)
    if args.loss_lpips:
        vgg_per = VGG16Perceptual()

    # Optimizer
    with torch.no_grad():
        f = model.forward_to(noise_vector, class_vector, args.truncation,
                args.num_feat_layer)
        output = model.forward_from(noise_vector, class_vector, args.truncation,
                f, args.num_feat_layer)
        output = output.add(1).div(2)
        writer.add_image('Initial', output.squeeze(0))
    f.requires_grad = True
    optimizer = optim.Adam([noise_vector])

    tbar = tqdm(range(args.iter))
    for i in tbar:
        output = model.forward_from(noise_vector, class_vector, args.truncation,
                f, args.num_feat_layer)
        output = output.add(1).div(2)

        # Loss
        loss = 0
        if args.loss_mse:
            loss_mse = nn.MSELoss()(target, output)
            loss += loss_mse
        if args.loss_lpips:
            loss_lpips = vgg_per.perceptual_loss(target, output)
            loss += loss_lpips

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tbar.set_postfix(loss=loss.item())

        if i % args.interval_save == 0:
            writer.add_image('recon', output.squeeze(0), i)
            writer.add_scalar('total', loss.item(), i)
            writer.add_scalar('mse', loss_mse.item(), i)
            if args.loss_lpips:
                writer.add_scalar('lpips', loss_lpips.item(), i)

    _, axs = plt.subplots(1, 2)

    writer.add_image('recon', output.squeeze(0), i + 1)


if __name__ == '__main__':
    args = parse()
    main(args)
