import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torchsummary import summary

import numpy as np
import scipy.io as scio
import time
import datetime
from tqdm import tqdm
from torch.autograd import Variable
import torch.autograd as autograd

from collections import OrderedDict

from dataset import MRI_dataset

from utils.config_reader import Config
# from train_options import Options
import train_options as config
from utils.utils import *

# import model.generator as models
import model.generator_unet4 as models

if __name__ == "__main__":
    # config = Config("./utils/params.yaml")
    # config = Options.parse()
    # print(config.BATCH_SIZE)
    
    # define GENERATOR and DISCRIMINATOR
    # net_layer = 4
    # num_features = 32
    net_layer = config.NUM_LAYERS
    num_features = config.NUM_FEATURES
    print(f"Model architecture is {net_layer} layers, with {num_features} intiial feature channels")
    if net_layer == 4:
        import model.generator_unet4 as models
    if net_layer == 3:
        import model.generator_unet3 as models   
        
    gen = models.datasetGAN(input_channels=1, output_channels=1, ngf=num_features)
    # summary(gen, (2, 128, 128))
    disc = models.Discriminator(in_channels=1, features=[32,64,128,256,512])
    
    if torch.cuda.device_count() > 1:
        gen = nn.DataParallel(gen) # DistributedDataParallel?
        disc = nn.DataParallel(disc)
    
    gen.to(config.DEVICE)
    disc.to(config.DEVICE)
        
    # initialize weights inside
    # gen.apply(initialize_weights)
    # disc.apply(initialize_weights)
    
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(config.B1, config.B2)) 
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(config.B1, config.B2))  
    
    # learning rate decay
    scheduler_gen = optim.lr_scheduler.LambdaLR(opt_gen, lr_lambda=lambda epoch: lambda_lr(epoch, config.LR_START_EPOCH, config.NUM_EPOCHS))
    scheduler_disc = optim.lr_scheduler.LambdaLR(opt_disc, lr_lambda=lambda epoch: lambda_lr(epoch, config.LR_START_EPOCH, config.NUM_EPOCHS))

    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()
    
    train_dataset = MRI_dataset(config, transform=None, train=True, test=False)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    # prev_time    = time.time()
    run_id = datetime.datetime.now().strftime("run_%H:%M:%S_%d/%m/%Y")
    
    save_config(config, run_id, config.SAVE_CHECKPOINT_DIR, config.RESULTS_DIR_NAME, train=True)
    
    for epoch in range(config.NUM_EPOCHS):
    # for epoch in range(5):
        
        loop = tqdm(train_loader, leave=True)
        epoch_losses = []
        for index, images in enumerate(loop):
            
            image_A, image_B, target_C = get_data_for_task(images, config.MODALITY_DIRECTION)
            
            image_A, image_B, target_C = image_A.to(config.DEVICE), image_B.to(config.DEVICE), target_C.to(config.DEVICE)
            
            x_concat = torch.cat((image_A, image_B), dim=1)
            target_fake, image_A_recon, image_B_recon = gen(x_concat)
            
            
            # backward of disc
            # -- Disc loss for fake --
            set_require_grad(disc, True)
            opt_disc.zero_grad()
            pred_disc_fake = disc(target_fake.detach()) # as dont want to backward this 

            loss_D_fake = criterion_GAN(pred_disc_fake, torch.zeros_like(pred_disc_fake))
            
            # -- Disc loss for real --
            pred_disc_fake = disc(target_C)
            loss_D_real = criterion_GAN(pred_disc_fake, torch.ones_like(pred_disc_fake))
            
            # get both loss and backprop
            loss_D = (loss_D_fake + loss_D_real) / 2
            loss_D.backward()
            opt_disc.step()
        
        
            # backward of gen
            set_require_grad(disc, False)
            opt_gen.zero_grad()
            
            pred_disc_fake = disc(target_fake) # D(x)
            
            # loss for GAN
            loss_G_BCE = criterion_GAN(pred_disc_fake, torch.ones_like(pred_disc_fake))
            loss_G_L1 = criterion_L1(target_fake, target_C) * config.LAMBDA_GAN_L1
            
            # loss for reconstucting unet
            loss_G_reconA = criterion_L1(image_A_recon, image_A)
            loss_G_reconB = criterion_L1(image_B_recon, image_B)
            
            # config.LAMBDA_BCE = 10
            # config.LAMBDA_RECON = 1
            
            # loss_G = 20*loss_G_BCE + 100*loss_G_L1 + 20*loss_G_reconA + 20*loss_G_reconB 
            loss_G = config.LAMBDA_BCE*loss_G_BCE + loss_G_L1 + config.LAMBDA_RECON*loss_G_reconA + config.LAMBDA_RECON*loss_G_reconB
            loss_G.backward()
            opt_gen.step()

            loop.set_description(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}]: Batch [{index+1}/{len(train_loader)}]")
            loop.set_postfix(loss_G=loss_G.item(), loss_D=loss_D.item())
            
            loss_dict = OrderedDict()
            
            batch_loss_dict = {
                "batch": index+1,
                "G_GAN": loss_G_BCE.item(),
                "G_L1": loss_G_L1.item(),
                "D_real": loss_D_real.item(),
                "D_fake": loss_D_fake.item(),
            }
            epoch_losses.append(batch_loss_dict)
        
        scheduler_disc.step()
        scheduler_gen.step()
         
        log_loss_to_json(config.SAVE_CHECKPOINT_DIR, config.RESULTS_DIR_NAME, run_id, epoch+1, epoch_losses)
        log_loss_to_txt(config.SAVE_CHECKPOINT_DIR, config.RESULTS_DIR_NAME, run_id, epoch+1, loss_G_BCE, loss_G_L1, loss_G_reconA, loss_G_reconB, loss_D_fake, loss_D_real)
           
        if config.SAVE_MODEL:
            if (epoch+1) > 5 and (epoch+1) % config.CHECKPOINT_INTERVAL == 0:
            # if (epoch+1) % config.CHECKPOINT_INTERVAL == 0:
                save_checkpoint(gen, opt_gen, checkpoint_dir=config.SAVE_CHECKPOINT_DIR, dir_name=config.RESULTS_DIR_NAME, save_filename=f"{epoch+1}_net_G.pth")
                save_checkpoint(disc, opt_disc, checkpoint_dir=config.SAVE_CHECKPOINT_DIR, dir_name=config.RESULTS_DIR_NAME, save_filename=f"{epoch+1}_net_D.pth")
                # print("checkpoint saved")
        
            

    


        
