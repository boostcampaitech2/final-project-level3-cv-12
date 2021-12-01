import os
import numpy as np
import random
import json
from tqdm import tqdm

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


def inference(pth_path, part):
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
    all_dataset = CEDataset('/opt/ml/project/data/all.json', part=part)
    
    # -- inference
    DB = []
    with torch.no_grad():
        model.eval()
        for i in tqdm(range(3964)):
            img, _ = all_dataset[i]
            img    = torch.Tensor(img).to(device)
            img    = img.unsqueeze(axis=0).float().to(device)
            img    = img.unsqueeze(axis=1).float().to(device)
            output = model(img)
            DB.append(output[0])
    return DB


def save_json(DB, save_path, file_name):
    res = {}
    for i, val in enumerate(DB):
        data = val.tolist()
        data = [round(i, 4) for i in data]
        res[str(i)] = data
    with open(save_path + file_name, 'w') as f:
        json.dump(res, f, indent=4)


if __name__ == '__main__':
    pth_path  = '/opt/ml/mouth_encoder.pth'
    save_path = '/opt/ml/project/inference/'
    part = 'mouth'
    DB = inference(pth_path, part)
    file_name = part + '.json'
    save_json(DB, save_path, file_name)
