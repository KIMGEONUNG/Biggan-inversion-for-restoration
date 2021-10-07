import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

__all__ = ['VGG16Perceptual', 'EncoderF']


class VGG16Perceptual():

    def __init__(self,
            resize=True,
            normalized_input=True,
            dev='cuda'):

        self.model = torch.hub.load('pytorch/vision:v0.8.2', 'vgg16',
                pretrained=True).to(dev).eval()

        self.normalized_intput = normalized_input
        self.dev = dev
        self.idx_targets = [1, 2, 13, 20]

        preprocess = []
        if resize:
            preprocess.append(transforms.Resize(256))
            preprocess.append(transforms.CenterCrop(224))
        if normalized_input:
            preprocess.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]))

        self.preprocess = transforms.Compose(preprocess)

    def get_mid_feats(self, x):
        x = self.preprocess(x)
        feats = []
        for i, layer in enumerate(self.model.features[:max(self.idx_targets) + 1]):
            x = layer(x)
            if i in self.idx_targets:
                feats.append(x)

        return feats

    def perceptual_loss(self, x1, x2):
        x1_feats = self.preprocess(x1)
        x2_feats = self.preprocess(x2)

        loss = 0
        for feat1, feat2 in zip(x1_feats, x2_feats):
            loss += feat1.sub(feat2).pow(2).mean()

        return loss / len(self.idx_targets)


class EncoderF(nn.Module):
    def __init__(self, nc=128, cls_ch=128, k_sz=3):
        super().__init__()
        self.layers = []
        self.input_conv = nn.Conv2d(3, 1 * nc, k_sz, 1, 1)  # 256, 256, 128
        self.block1 = SimpleBlockWithAGN(1 * nc, 2 * nc, cls_ch, k_sz)  # 128, 128, 256
        self.block2 = SimpleBlockWithAGN(2 * nc, 4 * nc, cls_ch, k_sz)  # 64,  64, 512
        self.block3 = SimpleBlockWithAGN(4 * nc, 8 * nc, cls_ch, k_sz)  # 32,  32, 1024
        self.block4 = SimpleBlockWithAGN(8 * nc, 8 * nc, cls_ch, k_sz)  # 16,  16, 1024

    def forward(self, x, c):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        x = F.relu(self.input_conv(x), True)  # 256, 256, 128
        x = F.avg_pool2d(x, [2, 2])          # 128, 128
        x = F.relu(self.block1(x, c), True)  # 128, 128, 256
        x = F.avg_pool2d(x, [2, 2])  # 64,  64
        x = F.relu(self.block2(x, c), True)  # 64,  64, 512
        x = F.avg_pool2d(x, [2, 2])  # 32,  32
        x = F.relu(self.block3(x, c), True)  # 32,  32, 1024
        x = F.avg_pool2d(x, [2, 2])  # 16,  16
        x = self.block4(x, c)  # 16,  16, 1024
        return x


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


class SimpleBlockWithAGN(nn.Module):
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


class SimpleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cls_ch, k_sz):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_ch, cls_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch // 2, k_sz, 1, 1)
        self.norm2 = nn.BatchNorm2d(out_ch // 2, cls_ch)
        self.conv2 = nn.Conv2d(out_ch // 2, out_ch, k_sz, 1, 1)
        if in_ch != out_ch:
            self.conv3 = nn.Conv2d(in_ch, out_ch, k_sz, 1, 1)
        return

    def forward(self, x):
        r = self.norm1(x)
        r = F.relu(r)
        r = self.conv1(r)
        r = self.norm2(r)
        r = F.relu(r, True)
        r = self.conv2(r)

        if hasattr(self, 'conv3'):
            x = self.conv3(x)
        return x + r


class EncoderF_Simple(nn.Module):
    def __init__(self, nc=128, cls_ch=128, k_sz=3):
        super().__init__()
        self.layers = []
        self.input_conv = nn.Conv2d(3, 1 * nc, k_sz, 1, 1)  # 256, 256, 128
        self.block1 = SimpleBlock(1 * nc, 2 * nc, cls_ch, k_sz)  # 128, 128, 256
        self.block2 = SimpleBlock(2 * nc, 4 * nc, cls_ch, k_sz)  # 64,  64, 512
        self.block3 = SimpleBlock(4 * nc, 8 * nc, cls_ch, k_sz)  # 32,  32, 1024
        self.block4 = SimpleBlock(8 * nc, 8 * nc, cls_ch, k_sz)  # 16,  16, 1024

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        x = F.relu(self.input_conv(x), True)  # 256, 256, 128
        x = F.avg_pool2d(x, [2, 2])          # 128, 128
        x = F.relu(self.block1(x), True)  # 128, 128, 256
        x = F.avg_pool2d(x, [2, 2])  # 64,  64
        x = F.relu(self.block2(x), True)  # 64,  64, 512
        x = F.avg_pool2d(x, [2, 2])  # 32,  32
        x = F.relu(self.block3(x), True)  # 32,  32, 1024
        x = F.avg_pool2d(x, [2, 2])  # 16,  16
        x = self.block4(x)  # 16, 1024
        return x
