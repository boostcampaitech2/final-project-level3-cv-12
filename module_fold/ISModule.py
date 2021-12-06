from module_fold.Block import Conv2D_Block, ResnetBlock, ConvTrans2D_Block
import torch
import torch.nn as nn
from module_fold.utils import weight_init_normal


class Generator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=56, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        super(Generator, self).__init__()
        activation = nn.ReLU()
        layers_list = []

        layers_list.append(nn.ReflectionPad2d(3))
        layers_list.append(Conv2D_Block(input_nc, ngf, 7, 0, 1))
        # downsample
        for i in range(n_downsampling):
            mult = (2 ** i)
            layers_list.append(Conv2D_Block(ngf*mult, ngf*mult*2, 4, 1, 2))

        # resnet blocks
        mult = (2 ** n_downsampling)
        for i in range(n_blocks):
            layers_list.append(ResnetBlock((ngf * mult), padding_type=padding_type,
                                           activation=activation, norm_layer=norm_layer))

        # upsample
        for i in range(n_downsampling):
            mult = (2 ** (n_downsampling - i))
            layers_list.append(ConvTrans2D_Block(
                ngf*mult, ngf*mult//2, 4, 1, 2))

        layers_list.append(nn.ReflectionPad2d(3))
        layers_list.append(nn.Conv2d(ngf, output_nc, 7, padding=0))
        layers_list.append(nn.Tanh())
        self.conv = nn.Sequential(*layers_list)

        for m in self.modules():
            weight_init_normal(m)

    def forward(self, input):
        return self.conv(input)


class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        def Dis_unit(in_channels):
            return nn.Sequential(
                Conv2D_Block(in_channels, 64, 4, 1, 2),
                Conv2D_Block(64, 128, 4, 1, 2),
                Conv2D_Block(128, 256, 4, 1, 2),
                Conv2D_Block(256, 512, 4, 0, 1),
                Conv2D_Block(512, 512, 4, 0, 1)
            )
        self.input_nc = input_nc
        self.dis = Dis_unit(self.input_nc+3)

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        return self.dis(img_input)
