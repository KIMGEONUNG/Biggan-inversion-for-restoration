import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
from PIL import Image
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_int,
        truncated_noise_sample, save_as_images, display_in_terminal)
import numpy as np
from tqdm import tqdm

# Class range 10 - 24 : brids


class AdaptiveGroupNorm(nn.Module):
    __constants__ = ['num_groups', 'num_channels', 'eps', 'affine', 'weight',
                     'bias']

    def __init__(self, num_channels, cond_dim, num_groups=None, eps=1e-5, affine=False):
        super(AdaptiveGroupNorm, self).__init__()
        self.num_channels = num_channels
        if num_groups is None:
            num_groups = 32
        self.num_groups = num_groups
        self.eps = eps
        self.affine = affine
        self.fc_gamma = nn.Linear(cond_dim, self.num_channels)
        self.fc_beta = nn.Linear(cond_dim, self.num_channels)
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(
                num_channels), requires_grad=True)
            self.bias = nn.Parameter(torch.Tensor(
                num_channels), requires_grad=True)
            nn.init.uniform_(self.weight)
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x, c):
        # Normalize
        x = F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)

        # Condition it
        gamma = 1.0 + self.fc_gamma(c)  # 1 centered
        beta = self.fc_beta(c)
        gamma = gamma.view(gamma.size(0), gamma.size(1), 1, 1)
        beta = beta.view(beta.size(0), beta.size(1), 1, 1)

        if self.affine:
            # learned affine (unnecessary since it can be learned but seems to help)
            weight = self.weight.view(1, -1, 1, 1).repeat(x.size(0), 1, 1, 1)
            bias = self.bias.view(1, -1, 1, 1).repeat(x.size(0), 1, 1, 1)
            gamma = gamma + weight
            beta = beta + bias

        return (gamma * x) + beta


class SimpleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cls_ch, k_sz):
        super().__init__()
        self.norm1 = AdaptiveGroupNorm(in_ch, cls_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch // 2, k_sz, 1, 1)
        self.norm2 = AdaptiveGroupNorm(out_ch // 2, cls_ch)
        self.conv2 = nn.Conv2d(out_ch // 2, out_ch, k_sz, 1, 1)
        if in_ch != out_ch:
            self.conv3 = nn.Conv2d(in_ch, out_ch, k_sz, 1, 1)
        return

    def forward(self, x, c):
        r = self.norm1(x, c)
        r = F.relu(r)
        r = self.conv1(r)

        r = self.norm2(r, c)
        r = F.relu(r, True)
        r = self.conv2(r)

        if hasattr(self, 'conv3'):
            x = self.conv3(x)
        return x + r


class SimpleEncoder(nn.Module):
    def __init__(self, nc, cls_ch, k_sz):
        super().__init__()
        self.conv = nn.Conv2d(3, 1 * nc, k_sz, 1, 1)  # 128 x 128
        self.block1 = SimpleBlock(1 * nc, 2 * nc, cls_ch, k_sz)  # 64 x 64
        self.block2 = SimpleBlock(2 * nc, 4 * nc, cls_ch, k_sz)  # 32 x 32
        self.block3 = SimpleBlock(4 * nc, 8 * nc, cls_ch, k_sz)  # 16 x 16
        self.block4 = SimpleBlock(8 * nc, 16 * nc, cls_ch, k_sz)  # 16 x 16
        self.block5 = SimpleBlock(16 * nc, 32 * nc, cls_ch, k_sz)  # 16 x 16
        return

    def forward(self, x, c):
        x = F.relu(self.conv(x), True)
        x = F.avg_pool2d(x, [2, 2])  # 128 x 128
        x = F.relu(self.block1(x, c), True)
        x = F.avg_pool2d(x, [2, 2])  # 64 x 64
        x = F.relu(self.block2(x, c), True)
        x = F.avg_pool2d(x, [2, 2])  # 32 x 32
        x = F.relu(self.block3(x, c), True)
        x = F.avg_pool2d(x, [2, 2])  # 16 x 16
        x = F.relu(self.block4(x, c), True)
        x = F.avg_pool2d(x, [2, 2])  # 8 x 8
        x = self.block5(x, c)
        return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = []
        self.resnet = SimpleEncoder(64, 128, 3)
        self.layer1 = SimpleBlock(2048, 2048, 128, 3)
        self.layer2 = SimpleBlock(2048, 2048, 128, 3)
        self.layer3 = SimpleBlock(2048, 2048, 128, 3)
        self.fc = nn.Linear(2048, 128)
        return

    def forward(self, x, c, st_idx=0, en_idx=5):
        """
        st_idx and en_idx is used to isolate layers.
        there are cleaner ways to do this
        """
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        assert en_idx <= 5
        assert st_idx >= 0
        if st_idx <= 0:
            x = self.resnet(x, c)

        if en_idx == 1:
            return x

        if st_idx <= 1:
            x = self.layer1(x, c)
        if en_idx == 2:
            return x

        if st_idx <= 2:
            x = F.avg_pool2d(x, [2, 2])
            x = self.layer2(x, c)
        if en_idx == 3:
            return x

        if st_idx <= 3:
            x = self.layer3(x, c)
        if en_idx == 4:
            return x

        if st_idx <= 4:
            x = F.adaptive_avg_pool2d(x, [1, 1])
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        return x


def sample():
    # Prepare a input
    truncation = 0.4
    size_batch = 16

    # for i in tqdm(range(893, 1000)):
    if True:
        i = 15
        class_vector = one_hot_from_int([i], batch_size=size_batch)
        # class_vector = one_hot_from_names(['soap bubble', 'coffee', 'mushroom'], batch_size=3)
        noise_vector = truncated_noise_sample(truncation=truncation, batch_size=size_batch)

        noise_vector = torch.from_numpy(noise_vector)
        class_vector = torch.from_numpy(class_vector)

        # If you have a GPU, put everything on cuda
        noise_vector = noise_vector.to('cuda')
        class_vector = class_vector.to('cuda')

        # name_model = 'biggan-deep-256'
        name_model = 'biggan-deep-512'
        model = BigGAN.from_pretrained(name_model)
        model.to('cuda')

        # Generate an image
        with torch.no_grad():
            output = model(noise_vector, class_vector, truncation)

        # If you have a GPU put back on CPU
        output = output.to('cpu')
        output = make_grid(output, int(size_batch ** 0.5), normalize=True)
        output = ToPILImage()(output)

        output.show()
        # output.save('./sample_%04d.png' % (i))

        exit()

if __name__ == '__main__':
    print('Program started')
    # sample()
    print('Program finished')
