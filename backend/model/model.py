import os
import json
import random
import pytz

import numpy as np
import torch
import torch.nn as nn
import cv2

from tqdm import tqdm
from datetime import datetime
from module_fold import CEModule, ISModule
from manifold.manifold import KNN, ConstrainedLeastSquareSolver


# ==========================================================
# -- basic function
# ==========================================================

part = ['left_eye', 'right_eye', 'nose', 'mouth', 'remainder']

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_time():
    d = datetime.now(pytz.timezone('Asia/Seoul'))
    return f'{d.month:0>2}{d.day:0>2}_{d.hour:0>2}{d.minute:0>2}'


# ==========================================================
# -- helper function _ get models
# ==========================================================

def get_encoder(encoder_path, part, device):
    encoder = CEModule.define_part_encoder(part=part)
    checkpoint = torch.load(encoder_path)
    state_dict = checkpoint.state_dict()
    encoder.load_state_dict(state_dict)
    return encoder.to(device)


def get_decoder(decoder_path, part, device):
    decoder = CEModule.define_part_decoder(part=part)
    checkpoint = torch.load(decoder_path)
    state_dict = checkpoint.state_dict()
    decoder.load_state_dict(state_dict)
    return decoder.to(device)


def get_generator(generator_path, device):
    generator = ISModule.Generator_U(input_nc=5, output_nc=3, ngf=56,
                                     n_blocks=9, norm_layer=nn.BatchNorm2d, padding_type='reflect')
    checkpoint = torch.load(generator_path)
    state_dict = checkpoint.state_dict()
    generator.load_state_dict(state_dict)
    return generator.to(device)


def get_knn(fv_json_path):
    fv_json = json.load(open(fv_json_path, 'r'))
    fv = [np.array(fv_json[i], dtype=np.float32) for i in fv_json]
    return KNN(fv)


# ==========================================================
# -- helper function _ data processing
# ==========================================================

def fv_proj(fv, knn, least_square, device, t, k):
    v_in = np.array(fv.detach().cpu().view(-1), dtype=np.float32)
    v_k = knn(v_in, k)
    ret = least_square(v_in, v_k)
    ret = torch.FloatTensor(ret).to(device).unsqueeze(axis=0)
    return t * fv + (1 - t) * ret


def save_image(img, img_path):
    img = img.squeeze(axis=0) * 255
    img = img.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    cv2.imwrite(img_path, img)


# ==========================================================
# -- main function
# ==========================================================

pos = [(256, 256), (302, 256), (244, 326), (244, 186), (385, 256)]
sz  = [512, 160, 128, 128, 128]

def inference(img, encoder, decoder, generator, knn, least_square, device, t, k):
    img = torch.FloatTensor(img).to(device)
    img = img.unsqueeze(axis=0)
    img = img.unsqueeze(axis=1)
    fv = encoder(img).to(device)
    fv = fv_proj(fv, knn, least_square, device, t, k)
    img = decoder(fv)

    sketch = torch.FloatTensor(np.zeros((1, 5, 512, 512))).to(device)
    img_patch = [img[:, :, pos[i][0]-sz[i]//2:pos[i][0]+sz[i]//2, pos[i][1]-sz[i]//2:pos[i][1]+sz[i]//2].clone() for i in range(5)]
    for i in range(1, 5): img_patch[0][:, :, pos[i][0]-sz[i]//2:pos[i][0]+sz[i]//2, pos[i][1]-sz[i]//2:pos[i][1]+sz[i]//2] = 0
    for i in range(5): sketch[:, i, pos[i][0]-sz[i]//2:pos[i][0]+sz[i]//2, pos[i][1]-sz[i]//2:pos[i][1]+sz[i]//2] = img_patch[i]
    output = generator(sketch)
    return output