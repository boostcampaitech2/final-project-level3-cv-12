import os
import json
import random

import numpy as np
import torch
import torch.nn as nn
import cv2

from tqdm import tqdm
from datetime import datetime
from data_loader.dataset import FEDataset
from torch.utils.data import DataLoader
from module_fold import CEModule, ISModule
from manifold.manifold import KNN, ConstrainedLeastSquareSolver
from albumentations.augmentations.geometric.resize import Resize

# ==========================================================
# -- basic function
# ==========================================================

part = ['remainder', 'nose', 'right_eye', 'left_eye', 'mouth']


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_time():
    d = datetime.now()
    return f'{d.month:2}{d.day:2}_{(d.hour+9)%24:2}{d.minute:2}'


# ==========================================================
# -- helper function _ get models
# ==========================================================

def get_encoder(encoder_file_path):
    encoder = [CEModule.define_part_encoder(part=part[i]) for i in range(5)]
    checkpoint_encoder = [torch.load(os.path.join(
        encoder_file_path, part[i] + '.pth')) for i in range(5)]
    statedict_encoder = [checkpoint_encoder[i].state_dict() for i in range(5)]
    for i in range(5):
        encoder[i].load_state_dict(statedict_encoder[i])
    return encoder


def get_decoder(decoder_file_path):
    decoder = [CEModule.define_part_decoder(part=part[i]) for i in range(5)]
    checkpoint_decoder = [torch.load(os.path.join(
        decoder_file_path, part[i] + '.pth')) for i in range(5)]
    statedict_decoder = [checkpoint_decoder[i].state_dict() for i in range(5)]
    for i in range(5):
        decoder[i].load_state_dict(statedict_decoder[i])
    return decoder


def get_generator(generator_pth_path):
    generator = ISModule.Generator_U(input_nc=5, output_nc=3, ngf=56,
                                     n_blocks=9, norm_layer=nn.BatchNorm2d, padding_type='reflect')
    checkpoint_generator = torch.load(generator_pth_path)
    statedict_generator = checkpoint_generator.state_dict()
    generator.load_state_dict(statedict_generator)
    return generator


def get_knn(fv_json_path):
    fv_json = json.load(open(fv_json_path, 'r'))
    fv_array = [[] for _ in range(5)]
    part = ['left_eye', 'right_eye', 'nose', 'mouth', 'remainder']
    for i in fv_json['left_eye']:
        for j, p in enumerate(part):
            fv_array[j].append(fv_json[p][i])
    fv_array = [np.array(i) for i in fv_array]
    knn = [KNN(fv_array[i]) for i in range(5)]
    return knn


# ==========================================================
# -- helper function _ data processing
# ==========================================================

def get_fv_array(fv_json, part):
    ret = [np.array(fv_json[part][i], dtype=np.float32) for i in fv_json]
    return ret


def apply_encoder(img, encoder, device):
    encoder.eval()
    img = torch.FloatTensor(img).to(device)
    img = img.unsqueeze(axis=0)
    img = img.unsqueeze(axis=1)
    return encoder(img)


def apply_decoder(fv, decoder, device):
    decoder.eval()
    img = torch.FloatTensor(fv).to(device)
    img = img.unsqueeze(axis=0)
    return decoder(img)


def fv_proj(fv, knn, least_square):
    v_in = np.array(fv, dtype=np.float32)
    v_k = knn(v_in)
    ret = least_square(v_in, v_k)
    return ret


# ==========================================================
# -- main function
# ==========================================================

def inference(img, generator, device):
    transform = Resize(height=512, width=512,
                       interpolation=1, always_apply=False, p=1)
    img = transform(image=img)['image']
    img = torch.FloatTensor(img).to(device)
    img = img.unsqueeze(axis=0)
    img = img.unsqueeze(axis=1)

    pos = [(256, 256), (256, 302), (326, 244), (186, 244), (256, 385)]
    sz = [512, 160, 128, 128, 128]
    img_part = [
        img[:, :, pos[0][1]-sz[0]//2:pos[0][1]+sz[0]//2,
            pos[0][0]-sz[0]//2:pos[0][0]+sz[0]//2].clone(),
        img[:, :, pos[1][1]-sz[1]//2:pos[1][1]+sz[1]//2,
            pos[1][0]-sz[1]//2:pos[1][0]+sz[1]//2].clone(),
        img[:, :, pos[2][1]-sz[2]//2:pos[2][1]+sz[2]//2,
            pos[2][0]-sz[2]//2:pos[2][0]+sz[2]//2].clone(),
        img[:, :, pos[3][1]-sz[3]//2:pos[3][1]+sz[3]//2,
            pos[3][0]-sz[3]//2:pos[3][0]+sz[3]//2].clone(),
        img[:, :, pos[4][1]-sz[4]//2:pos[4][1]+sz[4]//2,
            pos[4][0]-sz[4]//2:pos[4][0]+sz[4]//2].clone(),
    ]
    for i in range(1, 5):
        img_part[0][:, :, pos[i][1]-sz[i]//2:pos[i][1]+sz[i] //
                    2, pos[i][0]-sz[i]//2:pos[i][0]+sz[i]//2] = 0

    sketch = torch.FloatTensor(np.zeros((1, 5, 512, 512))).to(device)
    for i in range(5):
        sketch[:, i,
               pos[i][1]-sz[i]//2:pos[i][1]+sz[i]//2,
               pos[i][0]-sz[i]//2:pos[i][0]+sz[i]//2] = img_part[i]
    sketch.to(device)
    generator.to(device)
    output = generator(sketch)
    return output, sketch


def test(generator, device):
    save_img_path = f'/opt/ml/project/inference_save/{datetime.now().month}{get_time()}_image'
    load_img_path = '/opt/ml/project/inference_save/sketch'
    os.makedirs(save_img_path, exist_ok=True)
    print(save_img_path)

    for img_name in tqdm(sorted(os.listdir(load_img_path)[:100])):
        img_path = os.path.join(load_img_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(float)
        img = 1 - img/255
        output, _ = inference(img, generator, device)

        def SaveImage(img, name):
            img = img.squeeze(axis=0) * 255
            img = img.detach().cpu().numpy()
            img = np.transpose(img, (1, 2, 0))
            img_path = os.path.join(save_img_path, name + '_' + img_name)
            cv2.imwrite(img_path, img)
        SaveImage(output, 'output')
        # SaveImage(sketch, 'sketch')


if __name__ == '__main__':
    # -- setting
    seed_everything(42)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # -- file_path
    generator_pth_path = '/opt/ml/project/_DB/generator/generator.pth'

    # -- encoder, decoder, generator, knn, least_square
    generator = get_generator(generator_pth_path)

    # -- inference
    test(generator, device)
