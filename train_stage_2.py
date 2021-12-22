import os
import cv2
import random
from datetime import datetime
import pytz

import numpy as np
import multiprocessing
import wandb
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as A

from data_loader.dataset import FEDataset
from module_fold import ISModule
from math import log10


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
    d = datetime.now(pytz.timezone('Asia/Seoul'))
    return f'{d.month:0>2}{d.day:0>2}_{d.hour:0>2}{d.minute:0>2}'


def train(args):
    seed_everything(args.seed)
    run = wandb.init(project='Deep-drawing', entity='bcaitech_cv2')

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # -- dataset, data loader
    transform     = A.Compose([A.Cutout(always_apply=False, p=0.5, num_holes=8, max_h_size=8, max_w_size=8),
                               A.Blur(always_apply=False, p=0.5, blur_limit=(3, 4))])
    transform_all = A.Compose([A.RandomResizedCrop(512, 512, scale=(0.9, 1), ratio=(0.9, 1.12), always_apply=False, p = 0.5),
                               A.HorizontalFlip(always_apply=False, p=0.5)],
                               additional_targets={'image_trans': 'image'})
    train_dataset = FEDataset(args.train_json,
                              transform=transform,
                              transform_all=transform_all)
    val_dataset   = FEDataset(args.val_json)
    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size,
                               num_workers=multiprocessing.cpu_count()//2,
                               shuffle=True, drop_last=True)
    val_loader    = DataLoader(val_dataset, batch_size=1,
                               num_workers=multiprocessing.cpu_count()//2,
                               shuffle=False, drop_last=True)

    # -- model & loss & optimizer & scheduler
    generator = ISModule.Generator(input_nc=1, output_nc=3, ngf=56,
                                   n_downsampling=3, n_blocks=9,
                                   norm_layer=nn.BatchNorm2d,
                                   padding_type='reflect').to(device)
    discriminator = ISModule.Discriminator(input_nc=1).to(device)
    criterion_GAN = torch.nn.MSELoss()
    criterion_pixelwise = torch.nn.L1Loss()
    optimizer_G = torch.optim.AdamW(params=generator.parameters(), lr=0.0002, weight_decay=0.01)
    optimizer_D = torch.optim.AdamW(params=discriminator.parameters(), lr=0.0002, weight_decay=0.01)
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer_G, T_max=20)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer_D, T_max=20)

    lambda_pixel = 100
    columns = ['epoch', 'type', 'sketch', 'real', 'output']

    for epoch in range(args.epoch):
        test_table = wandb.Table(columns=columns)

        generator.train()
        discriminator.train()
        for step, (img, sketch) in enumerate(train_loader):
            img = np.transpose(img, (0, 3, 1, 2)).float().to(device)
            sketch = sketch.unsqueeze(axis=1).float().to(device)
            output = generator(sketch)
            discrim_fake = discriminator(output, sketch)
            loss_gan = criterion_GAN(discrim_fake, torch.ones_like(discrim_fake))
            loss_pixel = criterion_pixelwise(output, img)
            loss_G = loss_gan + loss_pixel * lambda_pixel
            optimizer_G.zero_grad()
            loss_G.backward(retain_graph=True)
            optimizer_G.step()

            discrim_real = discriminator(img, sketch)
            loss_real = criterion_GAN(discrim_real, torch.ones_like(discrim_real))
            discrim_fake = discriminator(output.detach(), sketch)
            loss_fake = criterion_GAN(discrim_fake, torch.zeros_like(discrim_fake))
            loss_D = (loss_fake + loss_real) / 2
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            wandb.log({'Train/loss_G': loss_G, 'Train/loss_D': loss_D})
            if (step + 1) % 10 == 0:
                sample_real = np.transpose(np.array(img[0].detach().cpu()), (1, 2, 0))
                sample_real = cv2.cvtColor(sample_real, cv2.COLOR_BGR2RGB)
                sample_fake = np.transpose(np.array(output[0].detach().cpu()), (1, 2, 0))
                sample_fake = cv2.cvtColor(sample_fake, cv2.COLOR_BGR2RGB)
                test_table.add_data(epoch+1, 'train',
                                    wandb.Image(sketch.squeeze(axis=1)[0]),
                                    wandb.Image(sample_real),
                                    wandb.Image(sample_fake))
                print(f'Epoch      : {epoch+1:>4}/{args.epoch:>4}',
                      f'Step       : {step+1:>4}/{len(train_loader):>4}',
                      f'D loss     : {loss_D.item():.6f}',
                      f'Gan loss   : {loss_gan.item():.6f}',
                      f'pixel loss : {loss_pixel.item():.6f}', sep='     ')
        scheduler_D.step()
        scheduler_G.step()

        with torch.no_grad():
            print('Calculating validation results...')
            generator.eval()
            discriminator.eval()
            total_psnr = 0.0
            for step, (img, sketch) in enumerate(val_loader):
                img = np.transpose(img, (0, 3, 1, 2)).float().to(device)
                sketch = sketch.unsqueeze(axis=1).float().to(device)
                output = generator(sketch)
                mse = criterion_GAN(output, img)
                psnr = 10 * log10(1 / mse.item())
                total_psnr += psnr
                print(f'Epoch    : {epoch+1:>4}/{args.epoch:>4}',
                      f'Step     : {step+1:>4}/{len(val_loader):>4}',
                      f'Avg PSNR : {round(total_psnr/(step+1), 2)}', sep='     ')
                sample_real = np.transpose(np.array(img[0].detach().cpu()), (1, 2, 0))
                sample_real = cv2.cvtColor(sample_real, cv2.COLOR_BGR2RGB)
                sample_fake = np.transpose(np.array(output[0].detach().cpu()), (1, 2, 0))
                sample_fake = cv2.cvtColor(sample_fake, cv2.COLOR_BGR2RGB)
                test_table.add_data(epoch+1, 'val',
                                    wandb.Image(sketch.squeeze(axis=1)[0]),
                                    wandb.Image(sample_real),
                                    wandb.Image(sample_fake))
            wandb.log({'Val/Average PSNR': round(total_psnr/len(val_loader), 2)})
            if (epoch + 1) % 10 == 0:
                save_model(generator, saved_dir=args.save_dir+'/generator', file_name=f'{epoch+1}_{get_time()}.pth')
                save_model(discriminator, saved_dir=args.save_dir+'/discriminator', file_name=f'{epoch+1}_{get_time()}.pth')
        run.log({'table_key': test_table})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_json', type=str, default='/opt/ml/project/data/aihub_train.json')
    parser.add_argument('--val_json',   type=str, default='/opt/ml/project/data/val.json')
    parser.add_argument('--save_dir',   type=str, default='/opt/ml/project/model_save')
    parser.add_argument('--seed',       type=int, default=42)
    parser.add_argument('--epoch',      type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=10)
    args = parser.parse_args()
    train(args)
