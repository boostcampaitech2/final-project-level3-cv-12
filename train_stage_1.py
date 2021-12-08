import argparse
import os

import numpy as np
import random
import multiprocessing
import cv2
import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import CEDataset
from module_fold import CEModule
from albumentations.augmentations.transforms import GaussNoise


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


def train(encoder, decoder, args):
    # -- settings
    seed_everything(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    run = wandb.init(project='Deep-drawing', entity='bcaitech_cv2')

    # -- dataset, data loader
    train_dataset = CEDataset('/opt/ml/project/data/train.json', args.part,
                              transform=GaussNoise(var_limit=(0, 1), mean=0.5, per_channel=True, always_apply=False, p=0.5))
    val_dataset  = CEDataset('/opt/ml/project/data/val.json', args.part)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=multiprocessing.cpu_count()//2,
                              shuffle=True,
                              pin_memory=use_cuda,
                              drop_last=True)
    val_loader   = DataLoader(val_dataset,
                              batch_size=args.batch_size,
                              num_workers=multiprocessing.cpu_count()//2,
                              shuffle=True,
                              pin_memory=use_cuda,
                              drop_last=True)

    #--- Loss & optimizer & scheduler
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

            test_table.add_data(epoch+1, 'train',
                                wandb.Image(t.squeeze(axis=1)[0]),
                                wandb.Image(x.squeeze(axis=1)[0]),
                                wandb.Image(y.squeeze(axis=1)[0]))
            wandb.log({'Train/loss': loss})
            wandb.log({'Train/lr': optimizer.param_groups[0]['lr']})

            if step % 50 == 0:
                print(f'Epoch : {epoch:>4}/{args.epoch:>4}',
                      f'Step  : {step:>4}/{len(train_loader):>4}',
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

                if step % 10 == 0:
                    path = f'{args.sample_img}/{args.part}/'
                    if epoch == 0:
                        os.makedirs(path, exist_ok=True)
                        img = (np.array(t.detach().cpu()).squeeze(axis=1)[0]*255).astype('uint8')
                        cv2.imwrite(path+f'original_{step}.png', img)
                    img = (np.array(y.detach().cpu()).squeeze(axis=1)[0]*255).astype('uint8')
                    cv2.imwrite(path+f'{epoch+1}_{step}.png', img)
            
            avg /= len(val_loader)
            wandb.log({'Val/loss': avg})
            print(f'Validation #{epoch+1} Average Loss : {round(avg.item(),6)}')
            if epoch % 10 == 9:
                save_model(encoder, saved_dir=args.save_dir, file_name=f'encoder_{args.part}_{epoch}_{int(avg.item() * 100000)}')
                save_model(decoder, saved_dir=args.save_dir, file_name=f'decoder_{args.part}_{epoch}_{int(avg.item() * 100000)}')
        run.log({'table_key': test_table})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str, default='left_eye')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='/opt/ml/project/save_pth')
    parser.add_argument('--sample_img', type=str, default='/opt/ml/project/save_pth/sample_img')
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    part_encoder = CEModule.define_part_encoder(model=args.part, norm='instance', input_nc=1, latent_dim=512)
    part_decoder = CEModule.define_part_decoder(model=args.part, norm='instance', output_nc=1, latent_dim=512)

    train(part_encoder, part_decoder, args)
