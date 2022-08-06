import torch
from math import cos, pi

SUPPORTED_OPTIMIZER_NAMES = ("Adam", "AdamW", "SGD")


def init_optimizer(name, parameters, **kwargs):
    if name not in SUPPORTED_OPTIMIZER_NAMES:
        raise NotImplementedError
    params = filter(lambda p: p.requires_grad, parameters)
    optimizer = getattr(torch.optim, name)(params, **kwargs)
    return [optimizer]

# modified from https://github.com/thangvubk/SoftGroup/blob/a87940664d0af38a3f55cdac46f86fc83b029ecc/softgroup/util/utils.py#L55
def cosine_lr_decay(optimizer, base_lr, current_epoch, start_epoch, total_epochs, clip):
    if current_epoch < start_epoch:
        return
    for param_group in optimizer.param_groups:
        param_group['lr'] = clip + 0.5 * (base_lr - clip) * (1 + cos(pi * ((current_epoch - start_epoch) / (total_epochs - start_epoch))))