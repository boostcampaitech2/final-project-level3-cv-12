import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random
from module_fold import ISModule
import argparse
import os
from dataset import FEDataset2
import multiprocessing
import wandb
from math import log10, remainder
import cv2


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def save_model(model, saved_dir, file_name):
    check_point = {'net': model.state_dict()}
    os.makedirs(saved_dir, exist_ok=True)
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model, output_path)


def train(args):
    seed_everything(args.seed)
    run = wandb.init(project="Deep-drawing", entity="bcaitech_cv2")

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset, data loader -> 각 part 에 맞는 sketch를 잘라서 받아온다.
    train_dataset = FEDataset2(json_path="/opt/ml/project/data/train.json")
    val_dataset = FEDataset2(json_path="/opt/ml/project/data/val.json")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=multiprocessing.cpu_count()//2,
                              shuffle=True,
                              pin_memory=use_cuda,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=multiprocessing.cpu_count()//2,
                            shuffle=False,
                            pin_memory=use_cuda,
                            drop_last=True)

    #--- Loss & optimizer & scheduler
    generator = ISModule.Generator_U(
        input_nc=1, output_nc=3, ngf=56, n_blocks=9, norm_layer=nn.BatchNorm2d, padding_type='reflect')
    discriminator = ISModule.Discriminator(input_nc=1)
    generator.to(device)
    discriminator.to(device)

    criterion_GAN = torch.nn.MSELoss()
    criterion_pixelwise = torch.nn.L1Loss()

    optimizer_G = torch.optim.AdamW(
        params=generator.parameters(), lr=0.001, weight_decay=0.01)
    optimizer_D = torch.optim.AdamW(
        params=discriminator.parameters(), lr=0.001, weight_decay=0.01)
    shcheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer_G, T_max=20)
    shcheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer_D, T_max=20)

    lambda_pixel = 100
    columns = ["epoch", "mode", "sketch", "real", "output"]

    for epoch in range(args.epoch):
        test_table = wandb.Table(columns=columns)
        generator.train()
        discriminator.train()

        for step, (img, sketch) in enumerate(train_loader):
            img = np.transpose(img, (0, 3, 1, 2))
            img, sketch = img.float().to(device), sketch.unsqueeze(axis=1).float().to(device)

            output = generator(sketch)
            discrim_fake = discriminator(output, sketch)

            loss_gan = criterion_GAN(
                discrim_fake, torch.ones_like(discrim_fake))
            loss_pixel = criterion_pixelwise(output, img)
            loss_G = loss_gan + loss_pixel*lambda_pixel

            optimizer_G.zero_grad()
            loss_G.backward(retain_graph=True)
            optimizer_G.step()

            discrim_real = discriminator(img, sketch)
            loss_real = criterion_GAN(
                discrim_real, torch.ones_like(discrim_real))

            discrim_fake = discriminator(output.detach(), sketch)
            loss_fake = criterion_GAN(
                discrim_fake, torch.zeros_like(discrim_fake))

            loss_D = (loss_fake+loss_real)*0.5

            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            sample_real = np.transpose(
                np.array(img[0].detach().cpu()), (1, 2, 0))
            sample_real = cv2.cvtColor(sample_real, cv2.COLOR_BGR2RGB)
            sample_fake = np.transpose(
                np.array(output[0].detach().cpu()), (1, 2, 0))
            sample_fake = cv2.cvtColor(sample_fake, cv2.COLOR_BGR2RGB)
            test_table.add_data(
                epoch+1, "train", wandb.Image(sketch.squeeze(axis=1)[0]), wandb.Image(sample_real), wandb.Image(sample_fake))

            wandb.log({"Train/loss_G": loss_G, "Train/loss_D": loss_D})
            if (step + 1) % 25 == 0:
                print(
                    f"[Epoch {epoch+1}/{args.epoch}] , Step [{step+1}/{len(train_loader)}],[D loss: {loss_D.item():.6f}] [Gan loss: {loss_gan.item():.6f}] [pixel loss: {loss_pixel.item():.6f}]")

        shcheduler_D.step()
        shcheduler_G.step()

        with torch.no_grad():
            print("Calculating validation results...")
            generator.eval()
            discriminator.eval()

            total_psnr = 0.0
            for step, (img, sketch) in enumerate(val_loader):
                img = np.transpose(img, (0, 3, 1, 2))
                img, sketch = img.float().to(device), sketch.unsqueeze(axis=1).float().to(device)
                output = generator(sketch)
                mse = criterion_GAN(output, img)
                psnr = 10 * log10(1/mse.item())
                total_psnr += psnr

                wandb.log({"Val/Average PSNR": round(total_psnr/(step+1), 2)})
                if step % 25 == 0:
                    print(
                        f"[Epoch {epoch+1}/{args.epoch}] , Step [{step+1}/{len(val_loader)}], Average PSNR [{round(total_psnr/(step+1),2)}]")

                sample_real = np.transpose(
                    np.array(img[0].detach().cpu()), (1, 2, 0))
                sample_real = cv2.cvtColor(sample_real, cv2.COLOR_BGR2RGB)
                sample_fake = np.transpose(
                    np.array(output[0].detach().cpu()), (1, 2, 0))
                sample_fake = cv2.cvtColor(sample_fake, cv2.COLOR_BGR2RGB)
                test_table.add_data(
                    epoch+1, "val", wandb.Image(sketch.squeeze(axis=1)[0]), wandb.Image(sample_real), wandb.Image(sample_fake))

            if epoch % 20 == 19:
                save_model(generator, saved_dir=args.save_dir+"/generator",
                           file_name=f"{epoch+1}.pth")
                save_model(discriminator, saved_dir=args.save_dir+"/discriminator",
                           file_name=f"{epoch+1}.pth")

        run.log({"table_key": test_table})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=21, help="Fixing seed")
    parser.add_argument("--save_dir", type=str,
                        default="/opt/ml/project/module_pth", help="Loactaion to save pth")
    parser.add_argument("--epoch", type=int,
                        default=200, help="Number of epoch")
    parser.add_argument("--batch_size", type=int,
                        default=4, help="Size of batch")

    args = parser.parse_args()

    train(args)
