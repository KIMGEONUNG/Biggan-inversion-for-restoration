#!/usr/bin/env python3

import torch
from pytorch_pretrained_biggan import (BigGAN, truncated_noise_sample)
from tqdm import tqdm
import argparse
from utils import set_seed
import matplotlib.pyplot as plt


DEV = 'cuda'


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--resolution', type=str, default='256')
    parser.add_argument('--class_index', type=int, default=15)
    parser.add_argument('--num_iter', type=int, default=1000)
    parser.add_argument('--size_batch', type=int, default=200)
    parser.add_argument('--truncation', type=float, default=1.0)
    parser.add_argument('--show', action='store_true')
    return parser.parse_args()


def cal_all(args):
    print(args)
    set_seed(args.seed)

    # Load model
    name_model = 'biggan-deep-%s' % (args.resolution)
    generator = BigGAN.from_pretrained(name_model)
    generator.eval()
    generator.to(DEV)

    ns = []
    for i in tqdm(range(args.num_iter)):
        z = truncated_noise_sample(truncation=args.truncation,
                batch_size=args.size_batch)
        z = torch.from_numpy(z)
        ns.append(z)

    ns = torch.cat(ns, dim=1)
    ns = ns.view(-1)
    print(ns.mean())
    print(ns.std())
    
    # plt.hist(ns.tolist())
    # plt.show()


def cal_mean(args):
    print(args)

    set_seed(args.seed)

    # Load model
    name_model = 'biggan-deep-%s' % (args.resolution)
    generator = BigGAN.from_pretrained(name_model)
    generator.eval()
    generator.to(DEV)

    ns = []
    for i in tqdm(range(args.num_iter)):
        z = truncated_noise_sample(truncation=args.truncation,
                batch_size=args.size_batch)
        z = torch.from_numpy(z)
        ns.append(z)

    ns = torch.cat(ns, dim=0)
    mean = ns.mean(dim=0)

    if args.show:
        plt.hist(mean, bins=100)
        plt.show()

    plt.hist(mean.tolist())
    plt.show()

    print(mean)


def cal_mean_norm(args):
    print(args)

    set_seed(args.seed)

    # Load model
    name_model = 'biggan-deep-%s' % (args.resolution)
    generator = BigGAN.from_pretrained(name_model)
    generator.eval()
    generator.to(DEV)

    ns = []
    for i in tqdm(range(args.num_iter)):
        z = truncated_noise_sample(truncation=args.truncation,
                batch_size=args.size_batch)
        z = torch.from_numpy(z)
        n = z.norm(dim=-1).tolist()
        ns += n

    if args.show:
        plt.hist(ns, bins=100)
        plt.show()

    z_mean = torch.Tensor(ns).mean()
    print(z_mean)


if __name__ == '__main__':
    args = parse()
    cal_all(args)
