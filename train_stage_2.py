import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random
from module_fold import FMModule, ISModule
import argparse
import os
from dataset import FEDataset
import multiprocessing
import wandb
from math import log10
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
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model, output_path)


def train(args):
    seed_everything(args.seed)
    wandb.init(project="Deep-drawing", entity="bcaitech_cv2")

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset, data loader -> 각 part 에 맞는 sketch를 잘라서 받아온다.
    train_dataset = FEDataset(json_path, fv_path)
    val_dataset = FEDataset(json_path, fv_path)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=multiprocessing.cpu_count()//2,
                              shuffle=True,
                              pin_memory=use_cuda,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=multiprocessing.cpu_count()//2,
                            shuffle=False,
                            pin_memory=use_cuda,
                            drop_last=True)

    #--- Loss & optimizer & scheduler
    decoder_mouth = FMModule.FMModule(
        norm_layer="instance", image_size=192, output_nc2=32, latent_dim=512)
    decoder_l_eye = FMModule.FMModule(
        norm_layer='instance', image_size=128, output_nc=32, latent_dim=512)
    decoder_r_eye = FMModule.FMModule(
        norm_layer='instance', image_size=128, output_nc=32, latent_dim=512)
    decoder_nose = FMModule.FMModule(
        norm_layer='instance', image_size=160, output_nc=32, latent_dim=512)
    decoder_remainder = FMModule.FMModule(
        norm_layer='instance', image_size=512, output_nc=32, latent_dim=512)

    generator = ISModule.Generator(input_nc=32, output_nc=3, ngf=56, n_downsampling=3,
                                   n_blocks=9, norm_layer=nn.BatchNorm, padding_type='reflect')
    discriminator = ISModule.Discriminator(input_nc=32)

    criterion_GAN = torch.nn.MSELoss()
    criterion_pixelwise = torch.nn.L1Loss()

    optimizer_G = torch.optim.AdamW(
        params=generator.parameters(), lr=0.001, weight_decay=0.01)
    optimizer_G.add_param_group(decoder_mouth.parameters())
    optimizer_G.add_param_group(decoder_l_eye.parameters())
    optimizer_G.add_param_group(decoder_r_eye.parameters())
    optimizer_G.add_param_group(decoder_nose.parameters())
    optimizer_G.add_param_group(decoder_remainder.parameters())
    optimizer_D = torch.optim.AdamW(
        params=discriminator.parameters(), lr=0.001, weight_decay=0.01)
    shcheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer_G, T_max=20)
    shcheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer_D, T_max=20)

    lambda_pixel = 100

    decoder_mouth.to(device)
    decoder_l_eye.to(device)
    decoder_r_eye.to(device)
    decoder_nose.to(device)
    decoder_remainder.to(device)
    generator.to(device)
    discriminator.to(device)

    for epoch in range(args.epoch):

        generator.train()
        discriminator.train()

        for step, (img, points, fvs) in enumerate(train_loader):
            img = img.to(device)
            whole_feature = decoder_remainder(fvs['remainder'])
            whole_feature[:, :, points['mouth'][1]-96: points['mouth'][1] + 96,
                          points['mouth'][0]-96:points['mouth'][0] + 96] = decoder_mouth(fvs['mouth'])
            whole_feature[:, :, points['nose'][1]-80: points['nose'][1] + 80,
                          points['nose'][0]-80: points['nose'][1]+80] = decoder_nose(fvs['nose'])
            whole_feature[:, :, points['left_eye'][1]-64: points['left_eye'][1] + 64,
                          points['left_eye'][0]-64:points['left_eye'][0] + 64] = decoder_l_eye(fvs['left_eye'])
            whole_feature[:, :, points['right_eye'][1]-64: points['right_eye'][1]+64,
                          points['right_eye'][0]-64: + points['right_eye'][0]+64] = decoder_r_eye(fvs['right_eye'])

            output = generator(whole_feature)
            discrim_fake = discriminator(output, whole_feature)

            loss_gan = criterion_GAN(
                discrim_fake, torch.ones_like(discrim_fake))
            loss_pixel = criterion_pixelwise(output, img)
            loss_G = loss_gan + loss_pixel*lambda_pixel

            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            discrim_real = discriminator(img, whole_feature)
            loss_real = criterion_GAN(
                discrim_real, torch.ones_like(discrim_real))

            discrim_fake = discriminator(output.detach(), whole_feature)
            loss_fake = criterion_GAN(
                discrim_fake, torch.zeros_like(discrim_fake))

            loss_D = (loss_fake+loss_real)*0.5

            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()
            print(
                f"[Epoch {epoch}/{n_epochs}] [D loss: {loss_D.item():.6f}] [G pixel loss: {loss_pixel.item():.6f}]")

        shcheduler_D.step()
        shcheduler_G.step()

        with torch.no_grad():
            print("Calculating validation results...")
            decoder_mouth.eval()
            decoder_l_eye.eval()
            decoder_r_eye.eval()
            decoder_nose.eval()
            decoder_remainder.eval()
            generator.eval()
            discriminator.eval()

            total_psnr = 0.0
            for step, (img, points, fvs) in enumerate(val_loader):

                whole_feature = decoder_remainder(fvs['remainder'])
                whole_feature[:, :, points['mouth'][1]-96: points['mouth'][1] + 96,
                              points['mouth'][0]-96:points['mouth'][0] + 96] = decoder_mouth(fvs['mouth'])
                whole_feature[:, :, points['nose'][1]-80: points['nose'][1] + 80,
                              points['nose'][0]-80: points['nose'][1]+80] = decoder_nose(fvs['nose'])
                whole_feature[:, :, points['left_eye'][1]-64: points['left_eye'][1] + 64,
                              points['left_eye'][0]-64:points['left_eye'][0] + 64] = decoder_l_eye(fvs['left_eye'])
                whole_feature[:, :, points['right_eye'][1]-64: points['right_eye'][1]+64,
                              points['right_eye'][0]-64: + points['right_eye'][0]+64] = decoder_r_eye(fvs['right_eye'])

                output = generator(whole_feature)
                mse = criterion_GAN(output, img)
                psnr = 10 * log10(1/mse.item())
                total_psnr += psnr

                sample_image = output[0]
                sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
                cv2.imwrite(, sample_image)
                print(f"Average PSNR is {round(total_psnr/(step+1),2)}")

            if epoch % 20 == 19:
                save_model(decoder_mouth, saved_dir=args.save_dir+"/feature_decoder_mouth",
                           file_name=f"{epoch+1}.pth")
                save_model(decoder_nose, saved_dir=args.save_dir+"/feature_decoder_nose",
                           file_name=f"{epoch+1}.pth")
                save_model(decoder_l_eye, saved_dir=args.save_dir+"/feature_decoder_l_eye",
                           file_name=f"{epoch+1}.pth")
                save_model(decoder_r_eye, saved_dir=args.save_dir+"/feature_decoder_r_eye",
                           file_name=f"{epoch+1}.pth")
                save_model(decoder_remainder, saved_dir=args.save_dir+"/feature_decoder_remainder",
                           file_name=f"{epoch+1}.pth")
                save_model(generator, saved_dir=args.save_dir+"/generator",
                           file_name=f"{epoch+1}.pth")
                save_model(discriminator, saved_dir=args.save_dir+"/discriminator",
                           file_name=f"{epoch+1}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=21, help="Fixing seed")
    parser.add_argument("--img_dir", type=str,
                        default="/opt/ml/project/final-project-level3-cv-12/images/images4x", help="Loactaion of Sketch")
    parser.add_argument("--sketch_dir", type=str,
                        default="/opt/ml/project/final-project-level3-cv-12/images/sketched_images", help="Loactaion of Sketch")
    parser.add_argument("--save_dir", type=str,
                        default="/opt/ml/project/final-project-level3-cv-12/module_pth", help="Loactaion to save pth")
    parser.add_argument("--sample_img_dir", type=str,
                        default="/opt/ml/project/final-project-level3-cv-12/sample_img", help="Loactaion to save sample image")
    parser.add_argument("--epoch", type=int,
                        default=200, help="Number of epoch")
    parser.add_argument("--batch_size", type=int,
                        default=8, help="Size of batch")

    args = parser.parse_args()

    train(args)
