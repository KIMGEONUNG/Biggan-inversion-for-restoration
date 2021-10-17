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
    parser.add_argument('--num_iter', type=int, default=100)
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
    parser.add_argument('--loss_mse', action='store_true', default=False)
    parser.add_argument('--loss_lpips', action='store_true', default=True)
    parser.add_argument('--loss_hsv', action='store_true', default=True)
    parser.add_argument('--loss_adv', action='store_true', default=False)

    # Loss coef
    parser.add_argument('--coef_mse', type=float, default=1.0)
    parser.add_argument('--coef_lpips', type=float, default=0.05)
    parser.add_argument('--coef_gen', type=float, default=0.05)
    parser.add_argument('--coef_hsv', type=float, default=1.0)
    return parser.parse_args()


def main(args):
    print(args)

    if args.seed >= 0:
        set_seed(args.seed)

    # Model
    name_model = 'biggan-deep-%s' % (args.resolution)
    generator = BigGAN.from_pretrained(name_model)
    generator.eval()
    generator.to(DEV)

    in_ch = 1
    encoder = EncoderF(in_ch).to(DEV)
    path ='./encoder_f.ckpt'

    encoder.load_state_dict(torch.load(path))
    encoder.eval()

    # Latents
    class_vector = one_hot_from_int([args.class_index],
            batch_size=args.size_batch)
    class_vector = torch.from_numpy(class_vector)
    class_vector = class_vector.to(DEV)

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
        im = ToPILImage()(grid_init)
        im.save('./z_effects/gt.jpg')
        if args.gray_inv:
            x_test = transforms.Grayscale()(x_test)

        for i in tqdm(range(args.num_iter)):

            x_test = x_test.to(DEV)

            z = truncated_noise_sample(truncation=args.truncation,
                    batch_size=args.size_batch)
            z = torch.from_numpy(z)
            z = z.to(DEV)

            f = encoder(x_test)
            output = generator.forward_from(z, class_vector,
                    args.truncation, f, args.num_feat_layer)
            output = output.add(1).div(2)

            grid = make_grid(output, nrow=4)
            im = ToPILImage()(grid)
            im.save('./z_effects/%03d.jpg' % (i))


if __name__ == '__main__':
    args = parse()
    main(args)
