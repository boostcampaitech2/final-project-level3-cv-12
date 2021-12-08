import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import cv2

from tqdm import tqdm
from module_fold.CEModule import define_part_encoder, define_part_decoder
from module_fold.ISModule import Generator
from manifold_projection.ManifoldProjection import KNN
from manifold_projection.ManifoldProjection import ConstrainedLeastSquareSolver


# ==========================================================
# basic function
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
# helper function _ get models
# ==========================================================

def get_encoder(encoder_file_path):
    encoder            = [define_part_encoder(model=part[i]) for i in range(5)]
    checkpoint_encoder = [torch.load(os.path.join(encoder_file_path, part[i] + '.pth')) for i in range(5)]
    statedict_encoder  = [checkpoint_encoder[i].state_dict() for i in range(5)]
    for i in range(5): encoder[i].load_state_dict(statedict_encoder[i])
    return encoder


def get_decoder(decoder_file_path):
    decoder            = [define_part_decoder(model=part[i]) for i in range(5)]
    checkpoint_decoder = [torch.load(os.path.join(decoder_file_path, part[i] + '.pth')) for i in range(5)]
    statedict_decoder  = [checkpoint_decoder[i].state_dict() for i in range(5)]
    for i in range(5): decoder[i].load_state_dict(statedict_decoder[i])
    return decoder


def get_generator(generator_pth_path):
    generator = Generator(input_nc=1, output_nc=3, ngf=56, n_downsampling=3,
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
# helper function _ data processing
# ==========================================================

def get_fv_array(fv_json, part):
    ret = []
    for i in fv_json:
        ret.append(np.array(fv_json[i][part], dtype=np.float32))
    return ret;


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
    fv   = fv.detach().cpu().view(-1)
    v_in = np.array(fv, dtype=np.float32)
    v_k  = knn(v_in)
    ret  = least_square(v_in, v_k)
    return ret


# ==========================================================
# main function
# ==========================================================

def inference(img, encoder, decoder, generator, knn, least_square, device):
    # get fv_vector using manifold projection
    pos  = {'left_eye' : (108, 156, 128),
            'right_eye': (255, 156, 128),
            'nose'     : (182, 232, 160),
            'mouth'    : (169, 301, 192),
            'remainder': (  0,   0, 512)}
    part_img = [img[pos[part[i]][1]:pos[part[i]][1]+pos[part[i]][2],
                    pos[part[i]][0]:pos[part[i]][0]+pos[part[i]][2]] for i in range(5)]
    fv = [apply_encoder(part_img[i], encoder[i], device) for i in range(5)]
    fv = [fv_proj(fv[i], knn[i], least_square) for i in range(5)]

    # FM Module
    fv = [apply_decoder(fv[i], decoder[i], device) for i in range(5)]
    whole_feature = np.zeros((1, 1, 512, 512))
    whole_feature = torch.FloatTensor(whole_feature).to(device)
    for i in range(4, -1, -1):
        whole_feature[:, 0:1,
                      pos[part[i]][1]:pos[part[i]][1]+pos[part[i]][2],
                      pos[part[i]][0]:pos[part[i]][0]+pos[part[i]][2]] = fv[i]

    # IS Module
    res = generator(whole_feature)
    return res, whole_feature


def last_inference(img, generator, device):
    img = torch.FloatTensor(img).to(device)
    img = img.unsqueeze(axis=0)
    img = img.unsqueeze(axis=1)
    return generator(img)


def test(encoder, decoder, generator, knn, least_square, device):
    load_img_path = '/opt/ml/project/test_inference/test_img'
    save_img_path = '/opt/ml/project/test_inference/save_img'
    for img_name in tqdm(sorted(os.listdir(load_img_path))):
        if 'png' not in img_name: continue
        img_path = os.path.join(load_img_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        img, wft = inference(img, encoder, decoder, generator, knn, least_square, device)
        img = img.squeeze(axis=0) * 255
        img = img.detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        cv2.imwrite(os.path.join(save_img_path, img_name), img)

        wft = wft.squeeze(axis=0) * 255
        wft = wft.detach().cpu().numpy()
        wft = np.transpose(wft, (1, 2, 0))
        cv2.imwrite(os.path.join(save_img_path, 'sketch'+img_name), wft)


def last_test(generator, device):
    load_img_path = '/opt/ml/project/test_inference/test_img'
    save_img_path = '/opt/ml/project/test_inference/save_img'
    for img_name in tqdm(sorted(os.listdir(load_img_path))):
        if 'png' not in img_name: continue
        img_path = os.path.join(load_img_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        img = last_inference(img, generator, device)
        img = img.squeeze(axis=0) * 255
        img = img.detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        cv2.imwrite(os.path.join(save_img_path, img_name), img)


if __name__ == "__main__":
    # setting
    seed_everything(42)
    use_cuda = torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")

    # file_path
    encoder_file_path  = '/opt/ml/project/encoder_pth'
    decoder_file_path  = '/opt/ml/project/decoder_pth'
    generator_pth_path = '/opt/ml/project/model_save/generator/40.pth'
    fv_json_path       = '/opt/ml/project/data/fv_train.json'

    # encoder, decoder, generator, knn, least_square
    encoder   = get_encoder(encoder_file_path)
    decoder   = get_decoder(decoder_file_path)
    generator = get_generator(generator_pth_path)
    for i in range(5):
        encoder[i].to(device)
        decoder[i].to(device)
    generator.to(device)
    knn = get_knn(fv_json_path)
    least_square = ConstrainedLeastSquareSolver()

    # inference
    # test(encoder, decoder, generator, knn, least_square, device)
    last_test(generator, device)
