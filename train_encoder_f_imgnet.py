#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToPILImage, ToTensor
import torchvision.transforms as transforms
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_int,
        truncated_noise_sample)
from tqdm import tqdm

import argparse
from torchvision.utils import make_grid

from model import VGG16Perceptual, EncoderF, DCGAN_D 
import numpy as np
import random

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from utils import set_seed, make_log_name, hsv_loss 


DEV = 'cuda'
IS_MULTIGPU = True


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2)
    # 5 --> 32, 4 --> 16, ...
    parser.add_argument('--num_feat_layer', type=int, default=4)
    parser.add_argument('--resolution', type=str, default='256')
    parser.add_argument('--num_epoch', type=int, default=1000)
    parser.add_argument('--interval_save', type=int, default=50)
    parser.add_argument('--size_batch', type=int, default=16)
    parser.add_argument('--truncation', type=float, default=1.0)
    parser.add_argument('--show', action='store_false')

    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

    # I/O
    parser.add_argument('--path_dataset', type=str, default='dataset_encoder')

    # Conversion
    parser.add_argument('--gray_inv', action='store_true', default=True)

    # Loss
    parser.add_argument('--loss_mse', action='store_true', default=True)
    parser.add_argument('--loss_lpips', action='store_true', default=False)
    parser.add_argument('--loss_hsv', action='store_true', default=False)

    # Loss coef
    parser.add_argument('--coef_mse', type=float, default=1.0)
    parser.add_argument('--coef_lpips', type=float, default=0.05)
    parser.add_argument('--coef_gen', type=float, default=0.05)
    parser.add_argument('--coef_hsv', type=float, default=1.0)
    return parser.parse_args()


def main(args):
    print(args)
    log_name = make_log_name(args, 'encoder_f_imgnet')

    if args.seed >= 0:
        set_seed(args.seed)

    # Logger
    path_log = 'runs/' + log_name
    writer = SummaryWriter(path_log)
    writer.add_text('config', str(args))
    print('logger name:', path_log)

    # Model
    name_model = 'biggan-deep-%s' % (args.resolution)
    biggan = BigGAN.from_pretrained(name_model)
    biggan.eval()
    biggan.to(DEV)
    biggan = nn.DataParallel(biggan)

    if args.loss_lpips:
        vgg_per = VGG16Perceptual()

    in_ch = 3
    if args.gray_inv:
        in_ch = 1

    encoder = EncoderF(in_ch).to(DEV)
    encoder.eval()
    encoder = nn.DataParallel(encoder)


    opt_target = []
    opt_target += list(encoder.parameters())

    # Optimizer
    optimizer_g = optim.Adam(opt_target, lr=args.lr, betas=(args.b1, args.b2))

    # Datasets
    prep = transforms.Compose([
            ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(256),
            ])
    dataset = ImageFolder(args.path_dataset, transform=prep)
    dataloader = DataLoader(dataset, batch_size=args.size_batch, shuffle=True,
            num_workers=8, drop_last=True)

    # Fix valid
    with torch.no_grad():
        x_test, x_test_class_index = next(iter(dataloader))
        grid_init = make_grid(x_test, nrow=4)
        writer.add_image('GT', grid_init)
        writer.flush()
        if args.gray_inv:
            x_test = transforms.Grayscale()(x_test)


        noise_vector_test = truncated_noise_sample(truncation=args.truncation,
                batch_size=args.size_batch)
        noise_vector_test = torch.from_numpy(noise_vector_test)
        noise_vector_test = noise_vector_test.to(DEV)

        x_test_class_index = one_hot_from_int(x_test_class_index, batch_size=args.size_batch)
        x_test_class_index = torch.from_numpy(x_test_class_index)
        x_test_class_index = x_test_class_index.to(DEV)
        
    truncation = torch.FloatTensor([args.truncation]).to(DEV)
    num_feat_layer = torch.IntTensor([args.num_feat_layer]).to(DEV)

    num_iter = 0
    for epoch in range(args.num_epoch):
        for i, (x, class_index) in enumerate(tqdm(dataloader)):
            num_iter += 1
            x = x.to(DEV)
            x_ = x.to(DEV)

            # Latents
            class_vector = one_hot_from_int(class_index, batch_size=args.size_batch)
            class_vector = torch.from_numpy(class_vector)
            class_vector = class_vector.to(DEV)

            if args.gray_inv:
                x_ = transforms.Grayscale()(x_)

            noise_vector = truncated_noise_sample(truncation=args.truncation,
                    batch_size=args.size_batch)
            noise_vector = torch.from_numpy(noise_vector)
            noise_vector = noise_vector.to(DEV)

            print(x_.shape)
            f = encoder(x_)
            print(f.shape)
            output = biggan(noise_vector, class_vector,
                    truncation, f, num_feat_layer)
            print(output.shape)
            exit()
            output = output.add(1).div(2)

            # Loss
            loss = 0
            loss_mse = torch.zeros(1)
            loss_lpips = torch.zeros(1)
            loss_hsv = torch.zeros(1)

            if args.loss_mse:
                loss_mse = args.coef_mse * nn.MSELoss()(x, output)
                loss += loss_mse
            if args.loss_hsv:
                loss_hsv = args.coef_hsv * hsv_loss(x, output)
                loss += loss_hsv
            if args.loss_lpips:
                loss_lpips = args.coef_lpips * vgg_per.perceptual_loss(x, output)
                loss += loss_lpips

            optimizer_g.zero_grad()
            loss.backward()
            optimizer_g.step()
            

            if i % args.interval_save == 0:
                writer.add_scalar('total_g', loss.item(), num_iter)
                writer.add_scalar('mse_rgb', loss_mse.item(), num_iter)
                writer.add_scalar('lpips', loss_lpips.item(), num_iter)
                writer.add_scalar('mse_hsv', loss_hsv.item(), num_iter)
                with torch.no_grad():
                    f = encoder(x_test.to(DEV))
                    output = biggan(noise_vector_test, x_test_class_index,
                            truncation, f, num_feat_layer)
                    output = output.add(1).div(2)
                    grid = make_grid(output, nrow=4)
                    writer.add_image('recon', grid, num_iter)
                    writer.flush()
                    torch.save(encoder.state_dict(), './encoder_f.ckpt') 


if __name__ == '__main__':
    args = parse()
    main(args)
