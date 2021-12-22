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
    d = datetime.now()
    return f'{d.month:2}{d.day:2}_{(d.hour+9)%24:2}{d.minute:2}'


# ==========================================================
# -- helper function _ get models
# ==========================================================

def get_encoder(encoder_file_path):
    encoder = CEModule.define_part_encoder(part='all')
    checkpoint = torch.load(encoder_file_path)
    state_dict = checkpoint.state_dict()
    encoder.load_state_dict(state_dict)
    return encoder


def get_decoder(decoder_file_path):
    decoder = CEModule.define_part_decoder(part='all')
    checkpoint = torch.load(decoder_file_path)
    state_dict = checkpoint.state_dict()
    decoder.load_state_dict(state_dict)
    return decoder


def get_generator(generator_pth_path):
    generator = ISModule.Generator(input_nc=1, output_nc=3, ngf=56, n_downsampling=3,
                                   n_blocks=9, norm_layer=nn.BatchNorm2d, padding_type='reflect')
    checkpoint_generator = torch.load(generator_pth_path)
    statedict_generator  = checkpoint_generator.state_dict()
    generator.load_state_dict(statedict_generator)
    return generator


def get_knn(fv_json_path):
    fv_json = json.load(open(fv_json_path, 'r'))
    knn = KNN(get_fv_array(fv_json))
    return knn


# ==========================================================
# -- helper function _ data processing
# ==========================================================

def get_fv_array(fv_json):
    ret = [np.array(fv_json[i], dtype=np.float32) for i in fv_json]
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

def inference(img, encoder, decoder, generator, knn, least_square, device):
    img = torch.FloatTensor(img).to(device)
    img = img.unsqueeze(axis=0)
    img = img.unsqueeze(axis=1)

    fv = encoder(img)
    def fv_proj(fv, t = 0.2):
        v_in = np.array(fv.detach().cpu().view(-1), dtype=np.float32)
        v_k = knn(v_in, 10)
        ret = least_square(v_in, v_k)
        ret = torch.FloatTensor(ret).to(device)
        ret = ret.unsqueeze(axis=0)
        return t * fv + (1 - t) * ret
    fv = fv_proj(fv)

    sketch = decoder(fv)
    output = generator(sketch)
    return output, sketch


def test(encoder, decoder, generator, knn, least_square, device):
    save_img_path = f'/opt/ml/project/inference_save/{datetime.now().month}{get_time()}_all'
    load_img_path = '/opt/ml/project/inference_save/sketch'
    os.makedirs(save_img_path, exist_ok=True)
    print(save_img_path)

    for img_name in tqdm(sorted(os.listdir(load_img_path)[:100])):
        img_path = os.path.join(load_img_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(float)
        img = 1 - img / 255
        output, sketch = inference(img, encoder, decoder, generator, knn, least_square, device)
        def SaveImage(img, name):
            img = img.squeeze(axis=0) * 255
            img = img.detach().cpu().numpy()
            img = np.transpose(img, (1, 2, 0))
            img_path = os.path.join(save_img_path, name + '_' + img_name)
            cv2.imwrite(img_path, img)
        SaveImage(output, 'output')
        SaveImage(sketch, 'sketch')


if __name__ == '__main__':
    # -- setting
    seed_everything(42)
    use_cuda = torch.cuda.is_available()
    device   = torch.device('cuda' if use_cuda else 'cpu')

    # -- file_path
    encoder_file_path  = '/opt/ml/project/model_save/encoder_all_260_1216_1424.pth'
    decoder_file_path  = '/opt/ml/project/model_save/decoder_all_260_1216_1424.pth'
    generator_pth_path = '/opt/ml/project/_DB/generator/generator_tj.pth'
    fv_json_path       = '/opt/ml/project/data/fv_all_train.json'

    # -- encoder, decoder, generator, knn, least_square
    encoder   = get_encoder(encoder_file_path)
    decoder   = get_decoder(decoder_file_path)
    generator = get_generator(generator_pth_path)
    encoder.to(device)
    decoder.to(device)
    generator.to(device)
    knn = get_knn(fv_json_path)
    least_square = ConstrainedLeastSquareSolver()

    # -- inference
    test(encoder, decoder, generator, knn, least_square, device)
