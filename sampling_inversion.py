#!/usr/bin/env python3

import torch
from torchvision.transforms import ToPILImage
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_int,
        truncated_noise_sample)
import argparse


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=str, default='256')
    parser.add_argument('--class_index', type=int, default=15)
    parser.add_argument('--truncation', type=float, default=0.4)
    return parser.parse_args()


def main(args):
    class_vector = one_hot_from_int([args.class_index],
            batch_size=1)
    # class_vector = one_hot_from_names(['soap bubble', 'coffee', 'mushroom'],
    # batch_size=3)
    noise_vector = truncated_noise_sample(truncation=args.truncation,
            batch_size=1)

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
    output = output.squeeze(0)
    output = output.add(1).div(2)
    output = ToPILImage()(output)
    output.save('./sample4inversion.jpg')


if __name__ == '__main__':
    args = parse()
    main(args)
