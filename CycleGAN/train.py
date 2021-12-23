import os
import numpy as np
import random
import torch
import torch.optim as optim
import argparse
import warnings
warnings.filterwarnings("ignore")
import time
import wandb

from dataset import get_data_loader
from model import CycleGAN
from utils import *
from losses import *


def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def train(args):
    
    seed_everything()
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    run = wandb.init(project = 'CycleGAN',entity='bcaitech_cv2')
    data_X_path = args.data_X_path
    data_Y_path = args.data_Y_path
    n_iters = args.n_iters
    batch_size = args.batch_size
    g_lr = args.g_lr
    d_lr = args.d_lr
    lambda_c = args.lambda_c
    sample_every = args.sample_every
    pretrained = args.pretrained
    pretrained_path = args.pretrained_path
    save_weights_dir = args.save_weights_dir
    sample_dir = args.sample_dir
    
    img_size = 128
    print_every=10
    
    
    
    #### Getting Data
    dataloader_X, validation_dataloader_X = get_data_loader(image_size=img_size,data_path=data_X_path,batch_size=batch_size)
    dataloader_Y, validation_dataloader_Y = get_data_loader(image_size=img_size,data_path=data_Y_path,batch_size=batch_size)
    valid_iter_X = iter(validation_dataloader_X)
    valid_iter_Y = iter(validation_dataloader_Y)
    
    G_XtoY, G_YtoX, D_X, D_Y = CycleGAN(pretrained=pretrained)

    if pretrained :
    # pretrained weights will be used for further training and evaluating the model.
        print("Using pretrained model..")
    
        D_X.load_state_dict(torch.load(pretrained_path+'/D_X5000.pth'))
        G_XtoY.load_state_dict(torch.load(pretrained_path+'/G_XtoY5000.pth'))
        D_Y.load_state_dict(torch.load(pretrained_path+'/D_Y2000.pth'))
        G_YtoX.load_state_dict(torch.load(pretrained_path+'/G_YtoX5000.pth'))
        
    
    # keep track of losses over time
    losses = []

    
    beta1=0.5
    beta2=0.999 # default value

    g_params = list(G_XtoY.parameters()) + list(G_YtoX.parameters())  # Get generator parameters

    # Create optimizers for the generators and discriminators
    g_optimizer = optim.Adam(g_params, g_lr, [beta1, beta2])
    d_x_optimizer = optim.Adam(D_X.parameters(), d_lr, [beta1, beta2])
    d_y_optimizer = optim.Adam(D_Y.parameters(), d_lr, [beta1, beta2])

    # Get some fixed data from domains X and Y for sampling. These are images that are held
    # constant throughout training, that allow us to inspect the model's performance.
    fixed_X = valid_iter_X.next()[0]
    fixed_Y = valid_iter_Y.next()[0]
    fixed_X = scale(fixed_X) # make sure to scale to a range -1 to 1
    fixed_Y = scale(fixed_Y)

    # batches per epoch
    iter_X = iter(dataloader_X)
    iter_Y = iter(dataloader_Y)
    batches_per_epoch = min(len(iter_X), len(iter_Y))
    
    print("Start Training...")

    for iters in range(1, n_iters+1):
        
        # test_table = wandb.Table(columns=columns)

        # Reset iterators for each epoch
        if iters % batches_per_epoch == 0:
            iter_X = iter(dataloader_X)
            iter_Y = iter(dataloader_Y)

        images_X, _ = iter_X.next()
        images_X = scale(images_X) # make sure to scale to a range -1 to 1

        images_Y, _ = iter_Y.next()
        images_Y = scale(images_Y)
        
        # move images to GPU if available (otherwise stay on CPU)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        images_X = images_X.to(device)
        images_Y = images_Y.to(device)


        # ============================================
        #            TRAIN THE DISCRIMINATORS
        # ============================================

        ##   First: D_X, real and fake loss components   ##
        # Train with real images
        d_x_optimizer.zero_grad()
        
        # 1. Compute the discriminator losses on real images  
        out_x = D_X(images_X)
        D_X_real_loss = real_mse_loss(out_x)
        
        # Train with fake images
        
        # 2. Generate fake images that look like domain X based on real images in domain Y
        fake_X = G_YtoX(images_Y)
        
        # 3. Compute the fake loss for D_X
        out_x = D_X(fake_X)
        D_X_fake_loss = fake_mse_loss(out_x)
        

        # 4. Compute the total loss and perform backprop
        d_x_loss = D_X_real_loss + D_X_fake_loss
        d_x_loss.backward()
        d_x_optimizer.step()


        ##   Second: D_Y, real and fake loss components   ##
        
        # Train with real images
        d_y_optimizer.zero_grad()
        
        # 1. Compute the discriminator losses on real images
        out_y = D_Y(images_Y)
        D_Y_real_loss = real_mse_loss(out_y)
        
        # Train with fake images
        # 2. Generate fake images that look like domain Y based on real images in domain X
        fake_Y = G_XtoY(images_X)

        # 3. Compute the fake loss for D_Y
        out_y = D_Y(fake_Y)
        D_Y_fake_loss = fake_mse_loss(out_y)

        # 4. Compute the total loss and perform backprop
        d_y_loss = D_Y_real_loss + D_Y_fake_loss
        d_y_loss.backward()
        d_y_optimizer.step()
        
        wandb.log({'Train/d_x_loss':d_x_loss})
        wandb.log({'Train/d_y_loss':d_y_loss})

        # =========================================
        #            TRAIN THE GENERATORS
        # =========================================

        ##    First: generate fake X images and reconstructed Y images    ##
        g_optimizer.zero_grad()

        # 1. Generate fake images that look like domain X based on real images in domain Y
        fake_X = G_YtoX(images_Y)

        # 2. Compute the generator loss based on domain X
        out_x = D_X(fake_X)
        g_YtoX_loss = real_mse_loss(out_x)

        # 3. Create a reconstructed y
        # 4. Compute the cycle consistency loss (the reconstruction loss)
        reconstructed_Y = G_XtoY(fake_X)
        reconstructed_y_loss = cycle_consistency_loss(images_Y, reconstructed_Y, lambda_weight=lambda_c)


        ##    Second: generate fake Y images and reconstructed X images    ##

        # 1. Generate fake images that look like domain Y based on real images in domain X
        fake_Y = G_XtoY(images_X)
        # 2. Compute the generator loss based on domain Y
        out_y = D_Y(fake_Y)
        g_XtoY_loss = real_mse_loss(out_y)

        # 3. Create a reconstructed x
        # 4. Compute the cycle consistency loss (the reconstruction loss)
        reconstructed_X = G_YtoX(fake_Y)
        reconstructed_x_loss = cycle_consistency_loss(images_X, reconstructed_X, lambda_weight=lambda_c)

        # 5. Add up all generator and reconstructed losses and perform backprop
        g_total_loss = g_YtoX_loss + g_XtoY_loss + reconstructed_y_loss + reconstructed_x_loss
        g_total_loss.backward()
        g_optimizer.step()
        
        wandb.log({'Train/g_total_loss':g_total_loss})


        # Print the log info
        if iters % print_every == 0:
            # append real and fake discriminator losses and the generator loss
            losses.append((d_x_loss.item(), d_y_loss.item(), g_total_loss.item()))
            t = time.strftime("%H:%M:%S")
            print(t+' - Iter [{:5d}/{:5d}] | d_X_loss: {:6.4f} | d_Y_loss: {:6.4f} | g_total_loss: {:6.4f}'.format(
                    iters, n_iters, d_x_loss.item(), d_y_loss.item(), g_total_loss.item()))

            
        # Save the generated samples
        if iters % sample_every == 0:
            G_YtoX.eval() # set generators to eval mode for sample generation
            G_XtoY.eval()
            save_samples(iters, fixed_Y, fixed_X, G_YtoX, G_XtoY, batch_size=batch_size,sample_dir=sample_dir)
            G_YtoX.train()
            G_XtoY.train()
            

        # uncomment these lines, if you want to save your model
        checkpoint_every=1000
        # Save the model parameters
        if iters % checkpoint_every == 0:
            save_weights(D_X, G_XtoY, D_Y, G_YtoX,iters,save_weights_dir=save_weights_dir)
            checkpoint(iters, G_XtoY, G_YtoX, D_X, D_Y)

    return losses

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_X_path", type=str, default="/opt/ml/CycleGAN/data/img")
    parser.add_argument("--data_Y_path", type=str, default="/opt/ml/CycleGAN/data/sketch")
    parser.add_argument("--n_iters", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--g_lr", type=float, default=0.0000001)
    parser.add_argument("--d_lr", type=float, default=0.0000001)
    # parser.add_argument("--lambda_d", type=float, default= 1.0)
    parser.add_argument("--lambda_c", type=float, default= 5.0)
    parser.add_argument("--sample_every", type=int, default=100)
    parser.add_argument("--pretrained", type=bool, default=True)
    parser.add_argument("--pretrained_path", type=str, default= "/opt/ml/CycleGAN/weights/thesis_2")
    parser.add_argument("--save_weights_dir", type=str, default = "/opt/ml/CycleGAN/weights/thesis_1")
    parser.add_argument("--sample_dir", type=str, default= "/opt/ml/CycleGAN/samples/thesis_1")

    args = parser.parse_args()
    print(args)
    


    train(args)
