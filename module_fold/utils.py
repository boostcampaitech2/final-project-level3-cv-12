import torch
import torch.nn as nn


def weight_init_normal(m):
    classname = m.__class__.__name__
    if 'Conv' in classname or 'Linear' in classname:
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif 'Norm' in classname:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0.0)


def weight_init_kaiming(m):
    classname = m.__class__.__name__
    if 'Conv' in classname or 'Linear' in classname:
        nn.init.kaiming_uniform_(m.weight)
    elif 'Norm' in classname:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0.0)
 