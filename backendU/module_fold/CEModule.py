import torch
import torch.nn as nn
from module_fold.Block import ResnetBlock, Conv2D_Block, ConvTrans2D_Block
from module_fold.utils import *


def define_part_encoder(part='mouth', norm='instance', input_nc=1, latent_dim=512):
    norm_layer = get_norm_layer(norm_type=norm)
    image_size = { 'left_eye': 128, 'right_eye': 128, 'nose': 160, 'mouth': 128, 'remainder': 512, 'all': 512 }
    net_encoder = CE_EncoderGen_Res(norm_layer, image_size[part], input_nc, latent_dim)
    print(f'net_encoder of part {part} is:', image_size[part])
    return net_encoder


def define_part_decoder(part='mouth', norm='instance', output_nc=1, latent_dim=512):
    norm_layer = get_norm_layer(norm_type=norm)
    image_size = { 'left_eye': 128, 'right_eye': 128, 'nose': 160, 'mouth': 128, 'remainder': 512, 'all': 512 }
    net_decoder = CE_DecoderGen_Res(norm_layer, image_size[part], output_nc, latent_dim)
    print(f'net_decoder to image of part {part} is:', image_size[part])
    return net_decoder


class CE_EncoderGen_Res(nn.Module):
    def __init__(self, norm_layer, image_size, input_nc, latent_dim=512):
        super(CE_EncoderGen_Res, self).__init__()
        self.norm_layer = norm_layer
        self.image_size = image_size
        self.input_nc = input_nc
        self.latent_dim = latent_dim

        latent_size = int(image_size / 32)
        longsize = 512 * latent_size ** 2
        self.longsize = longsize

        activation = nn.ReLU()
        padding_type = 'reflect'

        # conv1
        self.conv1_1 = Conv2D_Block(self.input_nc, 32, 4, 1, 2)
        self.conv1_2 = ResnetBlock(32, padding_type=padding_type, activation=activation, norm_layer=norm_layer)

        # conv2
        self.conv2_1 = Conv2D_Block(32, 64, 4, 1, 2)
        self.conv2_2 = ResnetBlock(64, padding_type=padding_type, activation=activation, norm_layer=norm_layer)

        # conv3
        self.conv3_1 = Conv2D_Block(64, 128, 4, 1, 2)
        self.conv3_2 = ResnetBlock(128, padding_type=padding_type, activation=activation, norm_layer=norm_layer)

        # conv4
        self.conv4_1 = Conv2D_Block(128, 256, 4, 1, 2)
        self.conv4_2 = ResnetBlock(256, padding_type=padding_type, activation=activation, norm_layer=norm_layer)

        # conv5
        self.conv5_1 = Conv2D_Block(256, 512, 4, 1, 2)
        self.conv5_2 = ResnetBlock(512, padding_type=padding_type, activation=activation, norm_layer=norm_layer)

        # fc
        self.fc = nn.Linear(in_features=longsize, out_features=latent_dim)

        for m in self.modules():
            weight_init_kaiming(m)

    def forward(self, x):
        h = self.conv1_1(x)
        h = self.conv1_2(h)
        h = self.conv2_1(h)
        h = self.conv2_2(h)
        h = self.conv3_1(h)
        h = self.conv3_2(h)
        h = self.conv4_1(h)
        h = self.conv4_2(h)
        h = self.conv5_1(h)
        h = self.conv5_2(h)
        h = h.view(h.shape[0], -1)
        return self.fc(h)


class CE_DecoderGen_Res(nn.Module):
    def __init__(self, norm_layer, image_size, output_nc, latent_dim=512):
        super(CE_DecoderGen_Res, self).__init__()
        self.norm_layer = norm_layer
        self.image_size = image_size
        self.input_nc = output_nc
        self.latent_dim = latent_dim

        latent_size = int(image_size / 32)
        self.latent_size = latent_size
        longsize = 512 * latent_size ** 2

        activation = nn.ReLU()
        padding_type = 'reflect'

        # fc
        self.fc = nn.Linear(in_features=latent_dim, out_features=longsize)

        # convTrans1
        self.convtr1_1 = ResnetBlock(512, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.convtr1_2 = ConvTrans2D_Block(512, 256, 4, 1, 2)

        # convTrans2
        self.convtr2_1 = ResnetBlock(256, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.convtr2_2 = ConvTrans2D_Block(256, 128, 4, 1, 2)

        # convTrans3
        self.convtr3_1 = ResnetBlock(128, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.convtr3_2 = ConvTrans2D_Block(128, 64, 4, 1, 2)

        # convTrans4
        self.convtr4_1 = ResnetBlock(64, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.convtr4_2 = ConvTrans2D_Block(64, 32, 4, 1, 2)

        # convTrans5
        self.convtr5_1 = ResnetBlock(32, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.convtr5_2 = ConvTrans2D_Block(32, 32, 4, 1, 2)

        # convTrans6
        self.convtr6_1 = ResnetBlock(32, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.convtr6_2 = nn.ReflectionPad2d(2)
        self.convtr6_3 = Conv2D_Block(32, output_nc, kernel_size=5, padding=0, stride=1)

        for m in self.modules():
            weight_init_kaiming(m)

    def forward(self, x):
        h = self.fc(x)
        h = h.view(h.shape[0], 512, self.latent_size, self.latent_size)
        h = self.convtr1_1(h)
        h = self.convtr1_2(h)
        h = self.convtr2_1(h)
        h = self.convtr2_2(h)
        h = self.convtr3_1(h)
        h = self.convtr3_2(h)
        h = self.convtr4_1(h)
        h = self.convtr4_2(h)
        h = self.convtr5_1(h)
        h = self.convtr5_2(h)
        h = self.convtr6_1(h)
        h = self.convtr6_2(h)
        return self.convtr6_3(h)
