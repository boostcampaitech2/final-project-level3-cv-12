import torch
import torch.nn as nn
from Block import ResnetBlock
from utils import weight_init_normal


class GlobalGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm, padding_type='reflect'):
        super(GlobalGenerator, self).__init__()
        activation = nn.ReLU()

        model = [nn.ReflectionPad2d(3), nn.Conv(
            input_nc, ngf, 7, padding=0), norm_layer(ngf), activation]
        # downsample
        for i in range(n_downsampling):
            mult = (2 ** i)
            model += [nn.Conv((ngf * mult), ((ngf * mult) * 2), 3, stride=2,
                              padding=1), norm_layer(((ngf * mult) * 2)), activation]

        # resnet blocks
        mult = (2 ** n_downsampling)
        for i in range(n_blocks):
            model += [ResnetBlock((ngf * mult), padding_type=padding_type,
                                  activation=activation, norm_layer=norm_layer)]

        # upsample
        for i in range(n_downsampling):
            mult = (2 ** (n_downsampling - i))
            model += [nn.ConvTranspose((ngf * mult), int(((ngf * mult) / 2)), 3, stride=2,
                                       padding=1, output_padding=1), norm_layer(int(((ngf * mult) / 2))), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv(ngf,
                                                 output_nc, 7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

        for m in self.modules():
            weight_init_normal(m)

    def execute(self, input):
        return self.model(input)
