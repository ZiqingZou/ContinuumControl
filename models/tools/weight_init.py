import torch.nn as nn


def apply_with_control(m: nn.Module, fn):
    do_recurse = fn(m)
    if not do_recurse:
        return
    for child in m.children():
        apply_with_control(child, fn)


def weight_init(m):
    if hasattr(m, 'reset_parameters') and callable(m.reset_parameters):
        m.reset_parameters()

        if getattr(m, 'is_logvar_head', False):
            nn.init.orthogonal_(m.weight, gain=0.01)
            nn.init.constant_(m.bias, -2.0)  # initial logvar = -2
            return False
        
        if getattr(m, 'is_mu_head', False):
            nn.init.orthogonal_(m.weight, gain=0.01)
            nn.init.constant_(m.bias, 0.0)
            return False

        return False  # do not recurse further
    return True
