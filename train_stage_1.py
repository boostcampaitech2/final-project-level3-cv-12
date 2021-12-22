import os
import cv2
import random

import numpy as np
import multiprocessing
import wandb
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as A

from datetime import datetime
from data_loader.dataset import CEDataset
from module_fold import CEModule


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def save_model(model, saved_dir, file_name):
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model, output_path)


def get_time():
    d = datetime.now()
    return f'{d.month:2}{d.day:2}_{(d.hour+9)%24:2}{d.minute:2}'


def train(args):
    # -- settings
    seed_everything(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    run = wandb.init(project='Deep-drawing', entity='bcaitech_cv2')

    # -- dataset, data loader
    sz = { 'left_eye': 128, 'right_eye': 128, 'nose': 160, 'mouth': 128, 'remainder': 512, 'all': 512 }
    transform     = A.Compose([A.Cutout(always_apply=False, p=0.5 if args.part == 'all' else 0.0, num_holes=8, max_h_size=8, max_w_size=8),
                               A.Blur(always_apply=False, p=0.5, blur_limit=(3, 4)),
                               A.GaussNoise(var_limit=(0, 0.2), mean=0.1, per_channel=True, always_apply=False, p=0.5)])
    transform_all = A.Compose([A.RandomResizedCrop(sz[args.part], sz[args.part], scale=(0.9, 1), ratio=(0.9, 1.12), always_apply=False, p = 0.5),
                               A.HorizontalFlip(always_apply=False, p=0.5 if 'eye' not in args.part else 0.0)],
                               additional_targets={'image_trans': 'image'})
    
    train_dataset = CEDataset(args.train_json, args.part,
                              transform=transform, transform_all=transform_all)
    val_dataset   = CEDataset(args.val_json, args.part)
    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size,
                               num_workers=multiprocessing.cpu_count()//2,
                               shuffle=True, pin_memory=use_cuda, drop_last=True)
    val_loader    = DataLoader(val_dataset, batch_size=1,
                               num_workers=multiprocessing.cpu_count()//2,
                               shuffle=False, pin_memory=use_cuda, drop_last=True)

    # -- model & loss & optimizer & scheduler
    encoder = CEModule.define_part_encoder(part=args.part)
    decoder = CEModule.define_part_decoder(part=args.part)
    model = nn.Sequential(encoder, decoder).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20)

    columns = ['epoch', 'mode', 'input', 'input with noise', 'output']

    for epoch in range(args.epoch):
        test_table = wandb.Table(columns=columns)

        model.train()
        for step, (t, x) in enumerate(train_loader):
            x = x.unsqueeze(axis=1).float().to(device)
            t = t.unsqueeze(axis=1).float().to(device)
            y = model(x)
            loss = criterion(y, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (step + 1) % 10 == 0:
                test_table.add_data(epoch+1, 'train',
                                    wandb.Image(t.squeeze(axis=1)[0]),
                                    wandb.Image(x.squeeze(axis=1)[0]),
                                    wandb.Image(y.squeeze(axis=1)[0]))
                wandb.log({'Train/loss': loss})
                wandb.log({'Train/lr': optimizer.param_groups[0]['lr']})
                print(f'Epoch : {epoch+1:>4}/{args.epoch:>4}',
                      f'Step  : {step+1:>4}/{len(train_loader):>4}',
                      f'Loss  : {round(loss.item(), 4)}', sep='     ')
        scheduler.step()

        with torch.no_grad():
            print('Calculating validation results...')
            model.eval()
            avg = 0.0
            for step, (t, x) in enumerate(val_loader):
                x = x.unsqueeze(axis=1).float().to(device)
                t = t.unsqueeze(axis=1).float().to(device)
                y = model(x)
                avg += criterion(y, t)
                test_table.add_data(epoch+1, 'val',
                                    wandb.Image(t.squeeze(axis=1)[0]),
                                    wandb.Image(x.squeeze(axis=1)[0]),
                                    wandb.Image(y.squeeze(axis=1)[0]))
            avg /= len(val_loader)
            wandb.log({'Val/loss': avg})
            print(f'Validation #{epoch+1} Average Loss : {round(avg.item(),6)}')
            if (epoch + 1) % 40 == 0:
                save_model(encoder, saved_dir=args.save_dir,
                           file_name=f'encoder_{args.part}_{epoch+1}_{get_time()}.pth')
                save_model(decoder, saved_dir=args.save_dir,
                           file_name=f'decoder_{args.part}_{epoch+1}_{get_time()}.pth')
        run.log({'table_key': test_table})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--part',       type=str, default='all')
    parser.add_argument('--train_json', type=str, default='/opt/ml/project/data/aihub_train.json')
    parser.add_argument('--val_json',   type=str, default='/opt/ml/project/data/val.json')
    parser.add_argument('--save_dir',   type=str, default='/opt/ml/project/model_save')
    parser.add_argument('--seed',       type=int, default=42)
    parser.add_argument('--epoch',      type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=20)
    args = parser.parse_args()
    train(args)
