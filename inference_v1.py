import os
import json
import random

import numpy as np
import torch
import torch.nn as nn
import cv2

from tqdm import tqdm
from data_loader.dataset import FEDataset
from torch.utils.data import DataLoader
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


# ==========================================================
# -- helper function _ get models
# ==========================================================

def get_encoder(encoder_file_path):
    encoder            = [CEModule.define_part_encoder(model=part[i]) for i in range(5)]
    checkpoint_encoder = [torch.load(os.path.join(encoder_file_path, part[i] + '.pth')) for i in range(5)]
    statedict_encoder  = [checkpoint_encoder[i].state_dict() for i in range(5)]
    for i in range(5): encoder[i].load_state_dict(statedict_encoder[i])
    return encoder


def get_decoder(decoder_file_path):
    decoder            = [CEModule.define_part_decoder(model=part[i]) for i in range(5)]
    checkpoint_decoder = [torch.load(os.path.join(decoder_file_path, part[i] + '.pth')) for i in range(5)]
    statedict_decoder  = [checkpoint_decoder[i].state_dict() for i in range(5)]
    for i in range(5): decoder[i].load_state_dict(statedict_decoder[i])
    return decoder


def get_generator(generator_pth_path):
    generator = ISModule.Generator(input_nc=5, output_nc=3, ngf=56, n_downsampling=3,
                          n_blocks=9, norm_layer=nn.BatchNorm2d, padding_type='reflect')
    checkpoint_generator = torch.load(generator_pth_path)
    statedict_generator  = checkpoint_generator.state_dict()
    generator.load_state_dict(statedict_generator)
    return generator


def get_knn(fv_json_path):
    fv_json = json.load(open(fv_json_path, 'r'))
    knn = [KNN(get_fv_array(fv_json, part[i])) for i in range(5)]
    return knn


# ==========================================================
# -- helper function _ data processing
# ==========================================================

def get_fv_array(fv_json, part):
    ret = [np.array(fv_json[i][part], dtype=np.float32) for i in fv_json]
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
    v_k  = knn(v_in)
    ret  = least_square(v_in, v_k)
    return ret


# ==========================================================
# -- main function
# ==========================================================

def inference(img, pos, encoder, decoder, generator, knn, least_square, device):
    img = img.to(device)

    sz = [128, 128, 160, 192, 512]
    img_part = [
        img[:, :, pos[0][1]-sz[0]//2:pos[0][1]+sz[0]//2, pos[0][0]-sz[0]//2:pos[0][0]+sz[0]//2],
        img[:, :, pos[1][1]-sz[1]//2:pos[1][1]+sz[1]//2, pos[1][0]-sz[1]//2:pos[1][0]+sz[1]//2],
        img[:, :, pos[2][1]-sz[2]//2:pos[2][1]+sz[2]//2, pos[2][0]-sz[2]//2:pos[2][0]+sz[2]//2],
        img[:, :, pos[3][1]-sz[3]//2:pos[3][1]+sz[3]//2, pos[3][0]-sz[3]//2:pos[3][0]+sz[3]//2],
        img[:, :, pos[4][1]-sz[4]//2:pos[4][1]+sz[4]//2, pos[4][0]-sz[4]//2:pos[4][0]+sz[4]//2],
    ]

    fv = [encoder[i](img_part[i]).to(device) for i in range(5)]
    
    def fv_proj(fv, idx, t = 0.5):
        v_in = np.array(fv.detach().cpu().view(-1), dtype=np.float32)
        v_k = knn[idx](v_in, 5)
        ret = least_square(v_in, v_k)
        ret = torch.FloatTensor(ret).to(device)
        ret = ret.unsqueeze(axis=0)
        return t * fv + (1 - t) * ret
    
    fv = [fv_proj(fv[i], i) for i in range(5)]

    whole_feature = torch.FloatTensor(np.zeros((1, 5, 512, 512))).to(device)
    for i in range(4, -1, -1):
        whole_feature[:, 4-i:5-i,
                      pos[i][1]-sz[i]//2:pos[i][1]+sz[i]//2,
                      pos[i][0]-sz[i]//2:pos[i][0]+sz[i]//2] = 1 - decoder[i](fv[i])

    sketch = torch.FloatTensor(np.zeros((1, 1, 512, 512))).to(device)
    for i in range(4, -1, -1):
        sketch[:, 0:1,
               pos[i][1]-sz[i]//2:pos[i][1]+sz[i]//2,
               pos[i][0]-sz[i]//2:pos[i][0]+sz[i]//2] = 1 - decoder[i](fv[i])

    output = generator(whole_feature)
    return output, sketch


def test(encoder, decoder, generator, knn, least_square, device):
    save_img_path = '/opt/ml/project/inference_save/v1'
    load_img_path = '/opt/ml/project/data/val.json'

    dataset = FEDataset(load_img_path)
    loader  = DataLoader(dataset)

    for i, (img, x, pos) in enumerate(tqdm(loader)):
        if i >= 10: break
        if i % 3 != 1: continue
        img = np.transpose(img, (0, 3, 1, 2)).float().to(device)
        x = x.unsqueeze(axis=1).float().to(device)
        output, y = inference(x, pos, encoder, decoder, generator, knn, least_square, device)
        def SaveImage(img, name):
            img = img.squeeze(axis=0) * 255
            img = img.detach().cpu().numpy()
            img = np.transpose(img, (1, 2, 0))
            img_path = os.path.join(save_img_path, name+str(i)+'.png')
            cv2.imwrite(img_path, img)
        SaveImage(img, 'input_image')
        SaveImage(output, 'output_image')
        SaveImage(1 - x, 'input_sketch')
        SaveImage(y, 'output_sketch')


if __name__ == '__main__':
    # -- setting
    seed_everything(42)
    use_cuda = torch.cuda.is_available()
    device   = torch.device('cuda' if use_cuda else 'cpu')

    # -- file_path
    encoder_file_path  = '/opt/ml/project/_DB/encoder'
    decoder_file_path  = '/opt/ml/project/_DB/decoder'
    generator_pth_path = '/opt/ml/project/_DB/generator/generator_80.pth'
    fv_json_path       = '/opt/ml/project/data/fv_train.json'

    # -- encoder, decoder, generator, knn, least_square
    encoder   = get_encoder(encoder_file_path)
    decoder   = get_decoder(decoder_file_path)
    generator = get_generator(generator_pth_path)
    for i in range(5):
        encoder[i].to(device)
        decoder[i].to(device)
    generator.to(device)
    knn = get_knn(fv_json_path)
    least_square = ConstrainedLeastSquareSolver()

    # -- inference
    test(encoder, decoder, generator, knn, least_square, device)
