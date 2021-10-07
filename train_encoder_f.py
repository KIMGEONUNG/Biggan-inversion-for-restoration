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

from model import VGG16Perceptual, EncoderF
import numpy as np
import random

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


DEV = 'cuda'


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=2)
    # 5 --> 32, 4 --> 16, ...
    parser.add_argument('--num_feat_layer', type=int, default=4)
    parser.add_argument('--resolution', type=str, default='256')
    parser.add_argument('--class_index', type=int, default=15)
    parser.add_argument('--num_epoch', type=int, default=30)
    parser.add_argument('--interval_save', type=int, default=3)
    parser.add_argument('--size_batch', type=int, default=16)
    parser.add_argument('--truncation', type=float, default=1.0)
    parser.add_argument('--show', action='store_false')

    # I/O
    parser.add_argument('--path_dataset', type=str, default='dataset_encoder')

    parser.add_argument('--opt_embd', action='store_true', default=False)

    # Loss
    parser.add_argument('--loss_mse', action='store_true', default=True)
    parser.add_argument('--loss_lpips', action='store_true', default=True)
    parser.add_argument('--coef_mse', type=float, default=1.0)
    parser.add_argument('--coef_lpips', type=float, default=0.05)
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
    if args.seed >= 0:
        set_seed(args.seed)

    # Logger
    path_log = 'runs/encoder_f'
    if args.loss_mse:
        path_log += '_mse%3.2f' % args.coef_mse
    if args.loss_lpips:
        path_log += '_lpips%3.2f' % args.coef_lpips
    if args.opt_embd:
        path_log += '_opt-embd'
    path_log += '_feat%02d' % args.num_feat_layer
    writer = SummaryWriter(path_log)
    writer.add_text('config', str(args))
    print('logger name:', path_log)

    # Model
    name_model = 'biggan-deep-%s' % (args.resolution)
    model = BigGAN.from_pretrained(name_model)
    model.to(DEV)
    if args.loss_lpips:
        vgg_per = VGG16Perceptual()
    encoder = EncoderF().to(DEV)

    # Latents
    class_vector = one_hot_from_int([args.class_index],
            batch_size=args.size_batch)
    class_vector = torch.from_numpy(class_vector)
    class_vector = class_vector.to(DEV)

    opt_target = []
    opt_target += list(encoder.parameters())

    if args.opt_embd:
        embd = Variable(model.embeddings(class_vector), requires_grad=True)
        opt_target += [embd]
    else:
        with torch.no_grad():
            embd = model.embeddings(class_vector)

    # Optimizer
    optimizer = optim.Adam(opt_target)

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

        noise_vector_test = truncated_noise_sample(truncation=args.truncation,
                batch_size=args.size_batch)
        noise_vector_test = torch.from_numpy(noise_vector_test)
        noise_vector_test = noise_vector_test.to(DEV)


    num_iter = 0
    for epoch in range(args.num_epoch):
        for i, (x, _) in enumerate(tqdm(dataloader)):
            num_iter += 1
            x = x.to(DEV)

            noise_vector = truncated_noise_sample(truncation=args.truncation,
                    batch_size=args.size_batch)
            noise_vector = torch.from_numpy(noise_vector)
            noise_vector = noise_vector.to(DEV)

            f = encoder(x, embd)
            output = model.forward_from(noise_vector, class_vector,
                    args.truncation, f, args.num_feat_layer)
            output = output.add(1).div(2)

            # Loss
            loss = 0
            loss_mse = torch.zeros(1)
            loss_lpips = torch.zeros(1)
            if args.loss_mse:
                loss_mse = args.coef_mse * nn.MSELoss()(x, output)
                loss += loss_mse
            if args.loss_lpips:
                loss_lpips = args.coef_lpips * vgg_per.perceptual_loss(x, output)
                loss += loss_lpips

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % args.interval_save == 0:
                writer.add_scalar('total', loss.item(), num_iter)
                writer.add_scalar('mse', loss_mse.item(), num_iter)
                writer.add_scalar('lpips', loss_lpips.item(), num_iter)
                with torch.no_grad():
                    f = encoder(x_test.to(DEV), embd)
                    output = model.forward_from(noise_vector_test, class_vector,
                            args.truncation, f, args.num_feat_layer)
                    output = output.add(1).div(2)
                    # print(output)
                    # print(output.min())
                    # print(output.max())
                    # print(output.mean())
                    # exit()
                    grid = make_grid(output, nrow=4)
                    writer.add_image('recon', grid, num_iter)
                    writer.flush()


if __name__ == '__main__':
    args = parse()
    main(args)
