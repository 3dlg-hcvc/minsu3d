import torch

SUPPORTED_OPTIMIZER_NAMES = ("Adam", "AdamW", "SGD")


def init_optimizer(name, parameters, **kwargs):
    if name not in SUPPORTED_OPTIMIZER_NAMES:
        raise NotImplementedError
    params = filter(lambda p: p.requires_grad, parameters)
    optimizer = getattr(torch.optim, name)(params, **kwargs)
    return [optimizer]
