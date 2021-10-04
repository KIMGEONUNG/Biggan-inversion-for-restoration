#!/usr/bin/env python3

import torch
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_int)

import argparse


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=str, default='256')
    parser.add_argument('--class_index', type=int, default=15)
    parser.add_argument('--size_batch', type=int, default=16)
    parser.add_argument('--truncation', type=float, default=0.8)
    parser.add_argument('--show', action='store_false')
    return parser.parse_args()


def truncated_noise_sample(batch_size=1, dim_z=128, truncation=1., seed=None):
    """ Create a truncated noise vector.
        Params:
            batch_size: batch size.
            dim_z: dimension of z
            truncation: truncation value to use
            seed: seed for the random generator
        Output:
            array of shape (batch_size, dim_z)
    """
    import numpy as np
    from scipy.stats import truncnorm

    values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=None).astype(np.float32)

    return truncation * values


def main(args):
    noise_vector = truncated_noise_sample(truncation=args.truncation,
            batch_size=args.size_batch)
    print(noise_vector)
    # noise_vector = torch.from_numpy(noise_vector)


if __name__ == '__main__':
    args = parse()
    main(args)
