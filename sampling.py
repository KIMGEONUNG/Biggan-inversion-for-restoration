#!/usr/bin/env python3

import torch
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_int,
        truncated_noise_sample)

import argparse


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=str, default='256')
    parser.add_argument('--class_index', type=int, default=15)
    parser.add_argument('--size_batch', type=int, default=16)
    parser.add_argument('--truncation', type=float, default=1.0)
    parser.add_argument('--show', action='store_false')
    return parser.parse_args()


def main(args):
    class_vector = one_hot_from_int([args.class_index],
            batch_size=args.size_batch)
    # class_vector = one_hot_from_names(['soap bubble', 'coffee', 'mushroom'],
    # batch_size=3)
    noise_vector = truncated_noise_sample(truncation=args.truncation,
            batch_size=args.size_batch)

    noise_vector = torch.from_numpy(noise_vector)
    class_vector = torch.from_numpy(class_vector)

    # If you have a GPU, put everything on cuda
    noise_vector = noise_vector.to('cuda')
    class_vector = class_vector.to('cuda')

    name_model = 'biggan-deep-%s' % (args.resolution)
    # name_model = 'biggan-deep-512'
    model = BigGAN.from_pretrained(name_model)
    model.to('cuda')

    # Generate an image
    with torch.no_grad():
        output = model(noise_vector, class_vector, args.truncation)

    # If you have a GPU put back on CPU
    output = output.to('cpu')
    output = make_grid(output, int(args.size_batch ** 0.5), normalize=True)
    output = ToPILImage()(output)

    if args.show:
        output.show()
    else:
        output.save('./sample_%04d.png' % (args.class_index))


if __name__ == '__main__':
    args = parse()
    main(args)
