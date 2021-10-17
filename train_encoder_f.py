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


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2)
    # 5 --> 32, 4 --> 16, ...
    parser.add_argument('--num_feat_layer', type=int, default=4)
    parser.add_argument('--resolution', type=str, default='256')
    parser.add_argument('--class_index', type=int, default=15)
    parser.add_argument('--num_epoch', type=int, default=400)
    parser.add_argument('--interval_save', type=int, default=3)
    parser.add_argument('--size_batch', type=int, default=16)
    parser.add_argument('--truncation', type=float, default=1.0)
    parser.add_argument('--show', action='store_false')

    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

    # I/O
    parser.add_argument('--path_dataset', type=str, default='dataset_encoder')
    parser.add_argument('--path_config', type=str, default='./checkpoints/config.pickle')
    parser.add_argument('--path_ckpt_D', type=str, default='./checkpoints/D_256.pth')

    # Conversion
    parser.add_argument('--gray_inv', action='store_true', default=True)

    # Loss
    parser.add_argument('--loss_mse', action='store_true', default=True)
    parser.add_argument('--loss_lpips', action='store_true', default=True)
    parser.add_argument('--loss_hsv', action='store_true', default=False)
    parser.add_argument('--loss_adv', action='store_true', default=False)

    # Loss coef
    parser.add_argument('--coef_mse', type=float, default=1.0)
    parser.add_argument('--coef_lpips', type=float, default=0.05)
    parser.add_argument('--coef_gen', type=float, default=0.05)
    parser.add_argument('--coef_hsv', type=float, default=1.0)
    return parser.parse_args()


def main(args):
    print(args)
    log_name = make_log_name(args, 'encoder_f')

    if args.seed >= 0:
        set_seed(args.seed)

    # Logger
    path_log = 'runs/' + log_name
    writer = SummaryWriter(path_log)
    writer.add_text('config', str(args))
    print('logger name:', path_log)

    # Model
    name_model = 'biggan-deep-%s' % (args.resolution)
    generator = BigGAN.from_pretrained(name_model)
    generator.eval()
    generator.to(DEV)
    if args.loss_lpips:
        vgg_per = VGG16Perceptual()
    if args.loss_adv:
        discriminator = DCGAN_D()
        discriminator.to(DEV)
        optimizer_d = optim.Adam(discriminator.parameters(),
                lr=args.lr, betas=(args.b1, args.b2))

    in_ch = 3
    if args.gray_inv:
        in_ch = 1

    encoder = EncoderF(in_ch).to(DEV)

    # Latents
    class_vector = one_hot_from_int([args.class_index],
            batch_size=args.size_batch)
    class_vector = torch.from_numpy(class_vector)
    class_vector = class_vector.to(DEV)

    opt_target = []
    opt_target += list(encoder.parameters())

    with torch.no_grad():
        embd = generator.embeddings(class_vector)

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
        x_test, _ = next(iter(dataloader))
        grid_init = make_grid(x_test, nrow=4)
        writer.add_image('GT', grid_init)
        writer.flush()
        if args.gray_inv:
            x_test = transforms.Grayscale()(x_test)

        noise_vector_test = truncated_noise_sample(truncation=args.truncation,
                batch_size=args.size_batch)
        noise_vector_test = torch.from_numpy(noise_vector_test)
        noise_vector_test = noise_vector_test.to(DEV)

    num_iter = 0
    for epoch in range(args.num_epoch):
        for i, (x, _) in enumerate(tqdm(dataloader)):
            num_iter += 1
            x = x.to(DEV)
            x_ = x.to(DEV)

            if args.gray_inv:
                x_ = transforms.Grayscale()(x_)

            noise_vector = truncated_noise_sample(truncation=args.truncation,
                    batch_size=args.size_batch)
            noise_vector = torch.from_numpy(noise_vector)
            noise_vector = noise_vector.to(DEV)

            f = encoder(x_)
            output = generator.forward_from(noise_vector, class_vector,
                    args.truncation, f, args.num_feat_layer)
            output = output.add(1).div(2)

            # Loss
            loss = 0
            loss_mse = torch.zeros(1)
            loss_lpips = torch.zeros(1)
            loss_hsv = torch.zeros(1)
            loss_adv = torch.zeros(1)
            loss_g = torch.zeros(1)
            loss_d = torch.zeros(1)

            if args.loss_mse:
                loss_mse = args.coef_mse * nn.MSELoss()(x, output)
                loss += loss_mse
            if args.loss_hsv:
                loss_hsv = args.coef_hsv * hsv_loss(x, output)
                loss += loss_hsv
            if args.loss_lpips:
                loss_lpips = args.coef_lpips * vgg_per.perceptual_loss(x, output)
                loss += loss_lpips

            if args.loss_adv:
                bce_fn = nn.BCELoss()
                real_label = 1.
                fake_label = 0.

                label = torch.full((args.size_batch,), real_label, 
                        dtype=torch.float).to(DEV)
                prop = discriminator(output.detach()).view(-1)
                loss_g = bce_fn(prop, label)
                loss_g = args.coef_gen * loss_g
                loss += loss_g

            optimizer_g.zero_grad()
            loss.backward()
            optimizer_g.step()
            
            # discriminator
            if args.loss_adv:
                label = torch.full((args.size_batch,), real_label, 
                        dtype=torch.float).to(DEV)
                prop = discriminator(x.detach()).view(-1)
                real_loss = bce_fn(prop, label)

                label = torch.full((args.size_batch,), fake_label, 
                        dtype=torch.float).to(DEV)
                prop = discriminator(output.detach()).view(-1)
                fake_loss = bce_fn(prop, label)

                loss_d = real_loss + fake_loss

                optimizer_d.zero_grad()
                loss_d.backward()
                optimizer_d.step()

            if i % args.interval_save == 0:
                writer.add_scalar('total_g', loss.item(), num_iter)
                writer.add_scalar('mse_rgb', loss_mse.item(), num_iter)
                writer.add_scalar('lpips', loss_lpips.item(), num_iter)
                writer.add_scalar('mse_hsv', loss_hsv.item(), num_iter)
                writer.add_scalar('generator', loss_g.item(), num_iter)
                writer.add_scalar('discriminator', loss_d.item(), num_iter)
                with torch.no_grad():
                    f = encoder(x_test.to(DEV))
                    output = generator.forward_from(noise_vector_test, class_vector,
                            args.truncation, f, args.num_feat_layer)
                    output = output.add(1).div(2)
                    grid = make_grid(output, nrow=4)
                    writer.add_image('recon', grid, num_iter)
                    writer.flush()
                    torch.save(encoder.state_dict(), './encoder_f.ckpt') 


if __name__ == '__main__':
    args = parse()
    main(args)
