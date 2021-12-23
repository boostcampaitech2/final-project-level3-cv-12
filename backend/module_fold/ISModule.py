import torch
import torch.nn as nn
from module_fold.Block import ResnetBlock, Conv2D_Block, ConvTrans2D_Block
from module_fold.utils import *


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=56, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        super(Generator, self).__init__()
        activation = nn.ReLU()
        layers_list = []
        layers_list.append(nn.ReflectionPad2d(3))
        layers_list.append(Conv2D_Block(input_nc, ngf, 7, 0, 1))

        # downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            layers_list.append(Conv2D_Block(ngf * mult, ngf * mult * 2, 4, 1, 2))

        # resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            layers_list.append(ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer))

        # upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            layers_list.append(ConvTrans2D_Block(ngf * mult, ngf * mult // 2, 4, 1, 2))

        layers_list.append(nn.ReflectionPad2d(3))
        layers_list.append(nn.Conv2d(ngf, output_nc, 7, padding=0))
        layers_list.append(nn.Tanh())
        self.conv = nn.Sequential(*layers_list)

        for m in self.modules():
            weight_init_kaiming(m)

    def forward(self, input):
        return self.conv(input)


class Generator_U(nn.Module):
    
    def __init__(self, input_nc, output_nc, ngf=56, n_blocks=9, norm_layer=nn.InstanceNorm2d, padding_type='reflect'):
        super(Generator_U, self).__init__()
        activation = nn.ReLU()

        first_layers = []
        first_layers.append(nn.ReflectionPad2d(3))
        first_layers.append(Conv2D_Block(input_nc, ngf, 7, 0, 1))
        self.first = nn.Sequential(*first_layers)

        # downsample
        self.down1 = Conv2D_Block(ngf, ngf * 2, 4, 1, 2)
        self.down2 = Conv2D_Block(ngf * 2, ngf * 4, 4, 1, 2)
        self.down3 = Conv2D_Block(ngf * 4, ngf * 8, 4, 1, 2)

        # resnet blocks
        resnet_layers = []
        for i in range(n_blocks):
            resnet_layers.append(ResnetBlock((ngf * 8), padding_type=padding_type, activation=activation, norm_layer=norm_layer))
        self.res = nn.Sequential(*resnet_layers)

        # upsample
        self.up3 = ConvTrans2D_Block(ngf * 8, ngf * 4, 4, 1, 2)
        self.up2 = ConvTrans2D_Block(ngf * 8, ngf * 2, 4, 1, 2)
        self.up1 = ConvTrans2D_Block(ngf * 4, ngf, 4, 1, 2)

        # final
        final_layers = []
        final_layers.append(nn.ReflectionPad2d(3))
        final_layers.append(nn.Conv2d(ngf, output_nc, 7, padding=0))
        final_layers.append(nn.Tanh())
        self.final = nn.Sequential(*final_layers)

        for m in self.modules():
            weight_init_kaiming(m)

    def forward(self, input):
        x = self.first(input)
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d3 = self.res(d3)
        u3 = self.up3(d3)
        u2 = self.up2(torch.cat((u3, d2), axis=1))
        u1 = self.up1(torch.cat((u2, d1), axis=1))
        return self.final(u1)


class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()
        self.input_nc = input_nc
        self.dis = nn.Sequential(
            Conv2D_Block(self.input_nc + 3, 64, 4, 1, 2),
            Conv2D_Block(64, 128, 4, 1, 2),
            Conv2D_Block(128, 256, 4, 1, 2),
            Conv2D_Block(256, 512, 4, 0, 1),
            Conv2D_Block(512, 512, 4, 0, 1)
        )

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        return self.dis(img_input)