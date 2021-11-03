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

from model import VGG16Perceptual, EncoderF, DCGAN_D, Discriminator_Fv3
import numpy as np
import random

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from utils import set_seed, make_log_name, hsv_loss 
import torch.autograd as autograd


DEV = 'cuda'


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2)
    # 5 --> 32, 4 --> 16, ...
    parser.add_argument('--num_feat_layer', type=int, default=4)
    parser.add_argument('--resolution', type=str, default='256')
    parser.add_argument('--class_index', type=int, default=15)
    parser.add_argument('--num_epoch', type=int, default=400)
    parser.add_argument('--num_iter', type=int, default=40000)
    parser.add_argument('--num_critic', type=int, default=5)
    parser.add_argument('--interval_save', type=int, default=3)
    parser.add_argument('--size_batch', type=int, default=16)
    parser.add_argument('--truncation', type=float, default=1.0)
    parser.add_argument('--show', action='store_false')

    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0., help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.9, help="adam: decay of first order momentum of gradient")

    parser.add_argument("--lr_d", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--b1_d", type=float, default=0., help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2_d", type=float, default=0.9, help="adam: decay of first order momentum of gradient")

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
    parser.add_argument('--loss_adv', action='store_true', default=True)

    # Loss coef
    parser.add_argument('--coef_mse', type=float, default=1.0)
    parser.add_argument('--coef_lpips', type=float, default=0.05)
    parser.add_argument('--coef_gen', type=float, default=0.5)
    parser.add_argument('--coef_hsv', type=float, default=1.0)
    parser.add_argument('--coef_adv', type=float, default=1.0)

    # Discriminator
    return parser.parse_args()


def calculate_gradient_penalty(D, real_images, fake_images, device,
        lambda_term=10):
    batch_size = real_images.size()[0]

    # eta
    eta = torch.FloatTensor(batch_size,1,1,1).uniform_(0,1)
    eta = eta.expand(batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
    eta = eta.to(device)

    interpolated = eta * real_images + ((1 - eta) * fake_images)
    interpolated = interpolated.to(device)

    # define it to calculate gradient
    interpolated = Variable(interpolated, requires_grad=True)

    # calculate probability of interpolated examples
    prob_interpolated = D(interpolated)

    # calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(outputs=prob_interpolated,
                              inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                           create_graph=True, retain_graph=True)[0]

    grad_penalty = ((gradients.norm(2, dim=[1, 2, 3]) - 1) ** 2) * lambda_term
    grad_penalty = grad_penalty.mean()
    return grad_penalty


def get_infinite_batches(data_loader):
    while True:
        for i, (images, _) in enumerate(data_loader):
            yield images


def main(args):
    print(args)
    log_name = make_log_name(args, 'encoder_f_adv_wgp')

    if args.seed >= 0:
        set_seed(args.seed)

    # Logger
    path_log = 'runs_adv_gp/' + log_name
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

    in_ch = 1

    encoder = EncoderF(in_ch).to(DEV)
    discriminator = Discriminator_Fv3().to(DEV)

    # Latents
    class_vector = one_hot_from_int([args.class_index],
            batch_size=args.size_batch)
    class_vector = torch.from_numpy(class_vector)
    class_vector = class_vector.to(DEV)

    opt_target = []
    opt_target += list(encoder.parameters())


    # Optimizer
    optimizer_g = optim.Adam(opt_target,
            lr=args.lr, betas=(args.b1, args.b2))
    optimizer_d = optim.Adam(discriminator.parameters(),
            lr=args.lr_d, betas=(args.b1_d, args.b2_d))

    # Datasets
    prep = transforms.Compose([
            ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(256),
            ])
    dataset = ImageFolder(args.path_dataset, transform=prep)
    dataloader = DataLoader(dataset, batch_size=args.size_batch, shuffle=True,
            num_workers=8, drop_last=True)
    dataloader = get_infinite_batches(dataloader)

    # Fix valid
    with torch.no_grad():
        x_test = dataloader.__next__()
        grid_init = make_grid(x_test, nrow=4)
        writer.add_image('GT', grid_init)
        writer.flush()
        if args.gray_inv:
            x_test = transforms.Grayscale()(x_test)

        noise_vector_test = truncated_noise_sample(truncation=args.truncation,
                batch_size=args.size_batch)
        noise_vector_test = torch.from_numpy(noise_vector_test)
        noise_vector_test = noise_vector_test.to(DEV)

    # Predefined Loss
    loss = 0
    loss_mse = torch.zeros(1)
    loss_lpips = torch.zeros(1)
    loss_hsv = torch.zeros(1)
    loss_adv = torch.zeros(1)
    loss_g = torch.zeros(1)
    loss_d = torch.zeros(1)
    prop_real = torch.ones(1) * 0.5
    prop_fake = torch.ones(1) * 0.5

    num_iter_real = 0
    for num_iter in tqdm(range(args.num_iter)):

        #########################
        ## DISCRIMINATOR START ##
        #########################
        # for p in discriminator.parameters():
        #     p.requires_grad = True  # to avoid computation

        for i in range(args.num_critic):
            num_iter_real += 1

            x = dataloader.__next__()
            x = x.to(DEV)
            x_ = x.to(DEV)
            x_ = transforms.Grayscale()(x_)

            optimizer_d.zero_grad()

            noise_vector = truncated_noise_sample(truncation=args.truncation,
                    batch_size=args.size_batch)
            noise_vector = torch.from_numpy(noise_vector)
            noise_vector = noise_vector.to(DEV)

            with torch.no_grad():
                f = encoder(x_) # [batch, 1024, 16, 16]
                f_real = generator.forward_to(noise_vector, class_vector,
                    args.truncation, args.num_feat_layer)

            score_fake_d = discriminator(f).mean()
            score_real_d = discriminator(f_real).mean()
            panelty = calculate_gradient_penalty( 
                    discriminator,
                    f_real.detach(), 
                    f.detach(), DEV)

            loss_d = -(score_real_d - score_fake_d) + panelty
            loss_d.backward()
            optimizer_d.step()

        #####################
        ## Generator Start ##
        #####################
        # for p in discriminator.parameters():
        #     p.requires_grad = False  # to avoid computation

        x = dataloader.__next__()
        x = x.to(DEV)
        x_ = x.to(DEV)
        x_ = transforms.Grayscale()(x_)

        optimizer_g.zero_grad()

        f = encoder(x_) # [batch, 1024, 16, 16]

        score_fake_g = discriminator(f.detach()).mean()
        loss_g = -score_fake_g
        loss_g = args.coef_gen * loss_g
        loss = loss_g 
        # loss_g.backward(retain_graph=True)

        output = generator.forward_from(noise_vector, class_vector,
                args.truncation, f, args.num_feat_layer)

        output = output.add(1).div(2)
        if args.loss_mse:
            loss_mse = args.coef_mse * nn.MSELoss()(x, output)
            loss += loss_mse
            # loss_mse.backward(retain_graph=True)
        if args.loss_hsv:
            loss_hsv = args.coef_hsv * hsv_loss(x, output)
            loss += loss_hsv
            # loss_hsv.backward(retain_graph=True)
        if args.loss_lpips:
            loss_lpips = args.coef_lpips * vgg_per.perceptual_loss(x, output)
            loss += loss_lpips
            # loss_lpips.backward(retain_graph=True)

        loss.backward()
        optimizer_g.step()
        #####################

        if num_iter % args.interval_save == 0:
            writer.add_scalar('mse_rgb', loss_mse.item(), num_iter_real)
            writer.add_scalar('lpips', loss_lpips.item(), num_iter_real)
            writer.add_scalars('critic',
                    {'D': loss_d.item(), 'G': loss_g.item()},
                    num_iter_real)
            with torch.no_grad():
                f = encoder(x_test.to(DEV))
                output = generator.forward_from(noise_vector_test, class_vector,
                        args.truncation, f, args.num_feat_layer)
                output = output.add(1).div(2)
                grid = make_grid(output, nrow=4)
                writer.add_image('recon', grid, num_iter_real)
                writer.flush()
                torch.save(encoder.state_dict(), './encoder_f_adv.ckpt') 


if __name__ == '__main__':
    args = parse()
    main(args)
