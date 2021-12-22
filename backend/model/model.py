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
    generator = ISModule.Generator(input_nc=1, output_nc=3, ngf=56, n_downsampling=3,
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

def inference(img, encoder, decoder, generator, knn, least_square, device, t, k):
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    img = torch.FloatTensor(img).to(device)
    img = img.unsqueeze(axis=0)
    img = img.unsqueeze(axis=1)
    fv = encoder(img).to(device)
    fv = fv_proj(fv, knn, least_square, device, t, k)
    sketch = decoder(fv)
    output = generator(sketch)
    save_image(output, "./output.png")


def test(encoder_path, decoder_path, generator_path, fv_json_path, t, k):
    # -- setting
    seed_everything(42)
    use_cuda = torch.cuda.is_available()
    device   = torch.device('cuda' if use_cuda else 'cpu')

    # -- encoder, decoder, generator, knn, least_square
    encoder = get_encoder(encoder_path, 'all', device)
    decoder = get_decoder(decoder_path, 'all', device)
    generator = get_generator(generator_path, device)
    knn = get_knn(fv_json_path)
    least_square = ConstrainedLeastSquareSolver()

    # -- img_path
    save_img_path = f'/opt/ml/project/inference_save/{get_time()}_all'
    load_img_path = '/opt/ml/project/inference_save/test'
    os.makedirs(save_img_path, exist_ok=True)
    print(save_img_path)

    for img_name in tqdm(sorted(os.listdir(load_img_path))):
        img_path = os.path.join(load_img_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(float)
        img = 1 - img / 255
        output, sketch = inference(img, encoder, decoder, generator, knn, least_square, device, t, k)
        save_image(output, os.path.join(save_img_path, 'output_' + img_name))
        save_image(1 - sketch, os.path.join(save_img_path, 'sketch_' + img_name))


if __name__ == '__main__':
    # -- file_path: left_eye, right_eye, nose, mouth, remainder
    encoder_path = '/opt/ml/project/model_save/encoder_all_260_1216_1424.pth'
    decoder_path = '/opt/ml/project/model_save/decoder_all_260_1216_1424.pth'
    generator_path = '/opt/ml/project/_DB/generator/generator_tj.pth'
    fv_json_path = '/opt/ml/project/data/fv_all_train.json'

    # -- parameter
    t = 0.4
    k = 5

    # -- inference
    test(encoder_path, decoder_path, generator_path, fv_json_path, t, k)