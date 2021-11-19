import torch
import torch.nn as nn
from Block import ResnetBlock, ConvTrans2D_Block
from utils import weight_init_normal


class FMModule(nn.Module):
    def __init__(self, norm_layer, image_size, output_nc, latent_dim=512):
        super(FMModule, self).__init__()

        latent_size = int(image_size/32)
        self.latent_size = latent_size
        longsize = 512*latent_size*latent_size

        activation = nn.ReLU()
        padding_type = 'reflect'
        norm_layer = nn.BatchNorm

        # Fully Connected
        self.fc = nn.Linear(in_features=latent_dim, out_features=longsize)

        # ConvTrans1
        self.convtr1_1 = ResnetBlock(
            512, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.convtr1_2 = ConvTrans2D_Block(512, 256, 4, 1, 2)

        # ConvTrans2
        self.convtr2_1 = ResnetBlock(
            256, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.convtr2_2 = ConvTrans2D_Block(256, 256, 4, 1, 2)

        # ConvTrans3
        self.convtr3_1 = ResnetBlock(
            256, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.convtr3_2 = ConvTrans2D_Block(256, 128, 4, 1, 2)

        # ConvTrans4
        self.convtr4_1 = ResnetBlock(
            128, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.convtr4_2 = ConvTrans2D_Block(128, 64, 4, 1, 2)

        # ConvTrans5
        self.convtr5_1 = ResnetBlock(
            64, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.convtr5_2 = ConvTrans2D_Block(64, 64, 4, 1, 2)

        # Lastlayer
        self.pad_1 = ResnetBlock(
            64, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.pad_2 = nn.ReflectionPad2d(2)

        self.convtr_last = ConvTrans2D_Block(64, output_nc, 5, 0, 2)

        for m in self.modules():
            weight_init_normal(m)

    def forward(self, x):
        h = self.convtr1_1(x)
        h = self.convtr1_2(h)

        h = self.convtr2_1(h)
        h = self.convtr2_2(h)

        h = self.convtr3_1(h)
        h = self.convtr3_2(h)

        h = self.convtr4_1(h)
        h = self.convtr4_2(h)

        h = self.convtr5_1(h)
        h = self.convtr5_2(h)

        h = self.pad_1(h)
        h = self.pad_2(h)

        return self.convtr_last(h)
