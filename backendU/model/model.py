import os
import json
import random
import pytz

import numpy as np
import torch
import torch.nn as nn
import cv2
from model.utils import *
from tqdm import tqdm
from datetime import datetime
from module_fold import CEModule
from module_fold.network import ResnetGenerator
from manifold.manifold import KNN, ConstrainedLeastSquareSolver
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
from model.dataset import ImageFolder



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
    generator = ResnetGenerator(input_nc=3, output_nc=3,n_blocks=4).to(device)
    state_dict = torch.load(generator_path)
    generator.load_state_dict(state_dict['genA2B'])
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
    save_image(img,'/opt/ml/backendv3/sketches/sk/sketch.png')

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    sketch_for_inference = ImageFolder('/opt/ml/backendv3/sketches', test_transform)
    inference_loader = DataLoader(sketch_for_inference, batch_size=1, shuffle=False)
    for n, (real_A, _) in enumerate(inference_loader):
        real_A = real_A.to(device)
        fake_A2B,_,_ = generator(real_A)
        output = np.concatenate((RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),), 0)
        output=cv2.resize(output, dsize=(512,512),interpolation=cv2.INTER_AREA)
        cv2.imwrite('output.png', output * 255.0)