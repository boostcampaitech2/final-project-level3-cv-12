import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random
from module_fold import FMModule, ISModule
import argparse
import os
from dataset import CustomVectorset
import multiprocessing
import wandb


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


def train(encoder, decoder, args):
    seed_everything(args.seed)
    run = wandb.init(project="Deep-drawing", entity="bcaitech_cv2")

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset, data loader -> 각 part 에 맞는 sketch를 잘라서 받아온다.
    train_dataset = CustomVectorset()

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=multiprocessing.cpu_count()//2,
                              shuffle=True,
                              pin_memory=use_cuda,
                              drop_last=True)
    #--- Loss & optimizer & scheduler
    decoder_mouth = FMModule.FMModule(
        norm_layer, image_size=192, output_nc=32, latent_dim=512)
    decoder_l_eye = FMModule.FMModule(
        norm_layer, image_size=128, output_nc=32, latent_dim=512)
    decoder_r_eye = FMModule.FMModule(
        norm_layer, image_size=128, output_nc=32, latent_dim=512)
    decoder_nose = FMModule.FMModule(
        norm_layer, image_size=160, output_nc=32, latent_dim=512)
    decoder_face = FMModule.FMModule(
        norm_layer, image_size=512, output_nc=32, latent_dim=512)

    generator = ISModule.GlobalGenerator()
    discriminator = ISModule.Discriminator()

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

    real = np.ones(512, 512)
    fake = np.zeros(512, 512)

    for epoch in range(args.epoch):

        generator.train()
        discriminator.train()

        for step, (mouth_v, l_eye_v, r_eye_v, nose_v, face_v, real_img) in enumerate(train_loader):
            whole_feature = decoder_face(face_v)
            whole_feature[:, :, 301:301 + 192,
                          169:169 + 192] = decoder_mouth(mouth_v)
            whole_feature[:, :, 232:232 + 160 - 36,
                          182:182 + 160] = decoder_nose(nose_v)
            whole_feature[:, :, 156:156 + 128,
                          108:108 + 128] = decoder_l_eye(l_eye_v)
            whole_feature[:, :, 156:156 + 128,
                          255:255 + 128] = decoder_r_eye(r_eye_v)

            output = generator(whole_feature)

            loss_gan = criterion_GAN(
                discriminator(output, whole_feature), real)
            loss_pixel = criterion_pixelwise(output, real_img)
            loss_G = loss_gan + loss_pixel

            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            loss_real = criterion_GAN(
                discriminator(real_img, whole_feature), real)
            loss_fake = criterion_GAN(discriminator(
                output.detach(), whole_feature), fake)
            loss_D = (loss_fake+loss_real)/2

            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

        shcheduler_D.step()
        shcheduler_G.step()

        with torch.no_grad():
            print("Calculating validation results...")
            generator.eval()
            discriminator.eval()

            for step, (mouth_v, l_eye_v, r_eye_v, nose_v, face_v, real_img) in enumerate(val_loader):
                whole_feature = decoder_face(face_v)
                whole_feature[:, :, 301:301 + 192,
                              169:169 + 192] = decoder_mouth(mouth_v)
                whole_feature[:, :, 232:232 + 160 - 36,
                              182:182 + 160] = decoder_nose(nose_v)
                whole_feature[:, :, 156:156 + 128,
                              108:108 + 128] = decoder_l_eye(l_eye_v)
                whole_feature[:, :, 156:156 + 128,
                              255:255 + 128] = decoder_r_eye(r_eye_v)

                output = generator(whole_feature)

                loss_gan = criterion_GAN(
                    discriminator(output, whole_feature), real)
                loss_pixel = criterion_pixelwise(output, real_img)
                loss_G = loss_gan + loss_pixel

                loss_real = criterion_GAN(
                    discriminator(real_img, whole_feature), real)
                loss_fake = criterion_GAN(discriminator(
                    output.detach(), whole_feature), fake)
                loss_D = (loss_fake+loss_real)/2

            save_model(generator, saved_dir=args.save_dir,
                       file_name=f"generator_latest.pth")


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
