import os
import numpy as np
import random
import json
from tqdm import tqdm
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from module_fold import CEModule
from dataset import CustomDataset
from data_loader.data_loader import CEDataset
import multiprocessing


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def test_encoder(pth_path, part, idx):
    # -- set model
    seed_everything(42)
    use_cuda = torch.cuda.is_available()
    device   = 'cuda' if use_cuda else 'cpu'

    model = CEModule.define_part_encoder(model=part)
    checkpoint = torch.load(pth_path)
    state_dict = checkpoint.state_dict()
    model.load_state_dict(state_dict)
    model = model.to(device)

    # -- dataset, data loader
    model.eval()
    all_dataset = CEDataset('/opt/ml/project/data/all.json', part=part)
    img, _ = all_dataset[0]
    img = torch.Tensor(img).to(device)
    img = torch.Tensor(all_dataset[idx][0])
    img = img.unsqueeze(axis=0).float().to(device)
    img = img.unsqueeze(axis=1).float().to(device)
    output = model(img)
    output = output[0].tolist()
    output = [round(i, 4) for i in output]
    return output


def test_decoder(pth_path, part, input):
    # -- set model
    seed_everything(42)
    use_cuda = torch.cuda.is_available()
    device   = 'cuda' if use_cuda else 'cpu'

    model = CEModule.define_part_decoder(model=part)
    checkpoint = torch.load(pth_path)
    state_dict = checkpoint.state_dict()
    model.load_state_dict(state_dict)
    model = model.to(device)

    # -- dataset, data loader
    model.eval()
    input = torch.Tensor(input)
    input = input.unsqueeze(axis=0).float().to(device)
    input = input.unsqueeze(axis=1).float().to(device)
    output = model(input)
    return output[0][0]


if __name__ == '__main__':
    pth_path  = '/opt/ml/mouth_encoder.pth'
    pth_path2 = '/opt/ml/mouth_decoder.pth'
    part = 'mouth'
    num = 0
    input = test_encoder(pth_path, part, num)
    output = test_decoder(pth_path2, part, input)
    data = (np.array(output.detach().cpu())*255).astype('uint8')
    cv2.imwrite(f'/opt/ml/project/inference/{num}.png', data)