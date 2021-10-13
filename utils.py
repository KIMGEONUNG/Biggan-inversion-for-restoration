import torch
import torch.nn as nn
import numpy as np
import random


__all__ = ['make_log_name', 'set_seed', 'hsv_loss']


def make_log_name(args, name):
    import regex
    ls = args._get_kwargs()
    for k, v in ls:
        if regex.search(r'^loss_', k):
            if v:
                name += '+%s:' % (k)
                key_target = 'coef_' + k[5:]
                coef = list(filter(lambda x: x[0] == key_target, ls))[0][1]
                name += '%4.3f' % (coef)
    return name


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def conversion(fn, x):
    if len(x.shape) == 4:
        x = x.permute(0, 2, 3, 1)
    else:
        x = x.permute(1, 2, 0)

    x = fn(x)

    if len(x.shape) == 4:
        x = x.permute(0, 3, 1, 2)
    else:
        x = x.permute(2, 0, 1)
    return x


def rgb2hsv_torch(rgb):
    # float and aleady normalized 

    arr = rgb
    out = torch.zeros_like(rgb)

    # -- V channel
    out_v, _ = arr.max(-1)

    # -- S channel
    delta = arr.max(-1).values - arr.min(-1).values

    out_s = delta / out_v
    out_s[delta == 0.] = 0.

    # -- H channel
    # red is max
    idx = (arr[..., 0] == out_v)
    out[..., 0][idx] = (arr[..., 1][idx] - arr[..., 2][idx]) / delta[idx]

    # green is max
    idx = (arr[..., 1] == out_v)
    out[..., 0][idx] = 2. + (arr[..., 2][idx] - arr[..., 0][idx]) / delta[idx]

    # blue is max
    idx = (arr[..., 2] == out_v)
    out[..., 0][idx] = 4. + (arr[..., 0][idx] - arr[..., 1][idx]) / delta[idx]
    out_h = (out[..., 0] / 6.) % 1.
    out_h[delta == 0.] = 0.

    out[..., 0] = out_h
    out[..., 1] = out_s
    out[..., 2] = out_v

    # # remove NaN
    out[torch.isnan(out)] = 0

    return out


def hsv_loss(x1, x2):
    x1_hsv = conversion(rgb2hsv_torch, x1)
    x2_hsv = conversion(rgb2hsv_torch, x2)
    return nn.MSELoss()(x1_hsv, x2_hsv)
