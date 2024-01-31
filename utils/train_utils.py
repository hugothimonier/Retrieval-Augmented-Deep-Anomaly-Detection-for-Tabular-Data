"""
Utils for model/optimizer initialization and training.
Forked from https://github.com/OATML/non-parametric-transformers
"""
import pprint

from torch import optim

from utils.optim_utils import Lookahead, Lamb

def init_optimizer(args, model_parameters):
    if 'default' in args.exp_optimizer:
        optimizer = optim.Adam(params=model_parameters, lr=args.exp_lr)
    elif 'lamb' in args.exp_optimizer:
        lamb = Lamb
        optimizer = lamb(
            model_parameters, lr=args.exp_lr, betas=(0.9, 0.999),
            weight_decay=args.exp_weight_decay, eps=1e-6)
    else:
        raise NotImplementedError

    if args.exp_optimizer.startswith('lookahead_'):
        optimizer = Lookahead(optimizer, k=args.exp_lookahead_update_cadence)

    return optimizer


def get_sorted_params(model):
    param_count_and_name = []
    for n,p in model.named_parameters():
        if p.requires_grad:
            param_count_and_name.append((p.numel(), n))

    pprint.pprint(sorted(param_count_and_name, reverse=True))


def count_parameters(model):
    r"""
    Due to Federico Baldassarre
    https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
