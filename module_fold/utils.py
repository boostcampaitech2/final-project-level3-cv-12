import torch
import torch.nn as nn


def get_norm_layer(norm_type='instance'):
    if (norm_type == 'batch'):
        norm_layer = nn.BatchNorm2d
    elif (norm_type == 'instance'):
        norm_layer = nn.InstanceNorm2d
    else:
        raise NotImplementedError(
            ('normalization layer [%s] is not found' % norm_type))
    return norm_layer


def weight_init_normal(m):
    classname = m.__class__.__name__
    if 'Conv' in classname or 'Linear' in classname:
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0.0)


def weight_init_kaiming(m):
    classname = m.__class__.__name__
    if 'Conv' in classname or 'Linear' in classname:
        nn.init.kaiming_uniform_(m.weight)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0.0)
