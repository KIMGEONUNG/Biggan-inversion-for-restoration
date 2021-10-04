#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_int,
        truncated_noise_sample)
from tqdm import tqdm
import matplotlib.pyplot as plt
import lpips

import os
import os.path
from os.path import join
import argparse

DEV = 'cuda' 

def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--resolution', type=str, default='256')
    parser.add_argument('--class_index', type=int, default=15)
    parser.add_argument('--iter', type=int, default=1000)
    parser.add_argument('--interval_save', type=int, default=10)
    parser.add_argument('--size_batch', type=int, default=1)
    parser.add_argument('--truncation', type=float, default=1.0)
    parser.add_argument('--show', action='store_false')

    # I/O
    parser.add_argument('--path_history', type=str, default='inversion_z_hist')
    parser.add_argument('--path_target', type=str, default='./real.jpg')

    # Loss
    parser.add_argument('--loss_mse', action='store_true', default=True)
    parser.add_argument('--loss_perceptual', action='store_true', default=True)
    return parser.parse_args()


def main(args):

    if not os.path.exists(args.path_history):
        os.mkdir(args.path_history)

    im = Image.open(args.path_target)
    target = ToTensor()(im).unsqueeze(0).to(DEV)

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
    if args.loss_perceptual:
        loss_fn_vgg = lpips.LPIPS(net='vgg').to(DEV)

    # Optimizer
    optimizer = optim.Adam([noise_vector])

    # Loss
    losses_mse = []
    losses_lpips = []
    losses = []

    tbar = tqdm(range(args.iter))
    for i in tbar:
        output = model(noise_vector, class_vector, args.truncation)
        output = output.add(1).div(2)

        # Loss
        loss = 0
        if args.loss_mse:
            loss_mse = nn.MSELoss()(target, output)
            loss += loss_mse
        if args.loss_perceptual:
            loss_lpips = loss_fn_vgg(target, output).sum()
            loss += loss_lpips

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tbar.set_postfix(loss=loss.item())

        if i % args.interval_save == 0:
            im_output = ToPILImage()(output.squeeze(0))
            path = join(args.path_history,
                    'inversion_z_%06d.jpg' % i)
            im_output.save(path)
            losses.append(loss.item())
            losses_mse.append(loss_mse.item())
            losses_lpips.append(loss_lpips.item())

    _, axs = plt.subplots(1, 2)

    im_output = ToPILImage()(output.squeeze(0))
    axs[0].imshow(im)
    axs[0].set_title('GT')
    axs[1].imshow(im_output)
    axs[1].set_title('Recon.')
    path = join(args.path_history, './inversion_z.jpg')
    plt.savefig(path)

    plt.clf()
    plt.plot(losses)
    plt.plot(losses_mse)
    plt.plot(losses_lpips)
    path = join(args.path_history, './plot_loss.jpg')
    plt.savefig(path)


if __name__ == '__main__':
    args = parse()
    main(args)
