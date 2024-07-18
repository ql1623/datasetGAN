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

from utils.config_reader import Config
# from train_options import Options
# import train_options as config
from utils.utils import *

from train_options_v2 import TrainOptions

if __name__ == "__main__":
    # config = Config("./utils/params.yaml")
    # config = Options.parse()
    # print(options.BATCH_SIZE)
    start_time = time.time()
    
    parser = TrainOptions()
    options = parser.parse()
    options.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(options.GPU_IDS)
     
    net_layer = options.NUM_LAYERS
    num_features = options.NUM_FEATURES
    print(f"Model architecture is {net_layer} layers, with {num_features} initial feature channels")
    
    if net_layer == 4:
        if options.CONDITION_METHOD == "concat":
            import model.cgan.generator_unet4_cgan as models
        elif options.CONDITION_METHOD == "add": # not concat but add
            import model.cgan.generator_unet4_cgan_v2 as models  
        else:  
            raise Exception("Condition method in GAN is not one of the predefined options")
    else:
        raise Exception("Number of UNET layers, net_layer is not specified to 4")
        
    num_modalities = 3   
    gan_version = options.GAN_VERSION
    gen = models.datasetGAN(input_channels=1, output_channels=1, ngf=num_features, version=gan_version)
    # summary(gen, (2, 128, 128))
    disc = models.Discriminator(in_channels=1, features=[32,64,128,256,512])
    
    if torch.cuda.device_count() > 1:
        gen = nn.DataParallel(gen, options.GPU_IDS) # DistributedDataParallel?
        disc = nn.DataParallel(disc, options.GPU_IDS)
    
    gen.to(options.DEVICE)
    disc.to(options.DEVICE)
        
    # initialize weights inside
    # gen.apply(initialize_weights)
    # disc.apply(initialize_weights)
    
    opt_disc = optim.Adam(disc.parameters(), lr=options.LEARNING_RATE, betas=(options.B1, options.B2)) 
    opt_gen = optim.Adam(gen.parameters(), lr=options.LEARNING_RATE, betas=(options.B1, options.B2))  
    
    # learning rate decay
    scheduler_gen = optim.lr_scheduler.LambdaLR(opt_gen, lr_lambda=lambda epoch: lambda_lr(epoch, options.LR_START_EPOCH, options.NUM_EPOCHS))
    scheduler_disc = optim.lr_scheduler.LambdaLR(opt_disc, lr_lambda=lambda epoch: lambda_lr(epoch, options.LR_START_EPOCH, options.NUM_EPOCHS))

    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()
    criterion_GDL = GradientDifferenceLoss()
    
    dataset_version = options.DATASET_VERSION
    if dataset_version == 1:
        from dataset_png import MRI_dataset
        train_dataset = MRI_dataset(config, transform=True, train=True)
    elif dataset_version == 2:
        from dataset_png_v2 import MRI_dataset
        train_dataset = MRI_dataset(config, transform=True, train=True)
    elif dataset_version == 3:
        from dataset_png_v3 import MRI_dataset  
        train_dataset = MRI_dataset(options.INPUT_MODALITIES, options.DATA_PNG_DIR, transform=True, train=True)
        
    train_loader = DataLoader(train_dataset, batch_size=options.BATCH_SIZE, shuffle=True, num_workers=options.NUM_WORKERS)
    
    run_id = datetime.datetime.now().strftime("run_%H:%M:%S_%d/%m/%Y")
    parser.save_options(run_id, options.SAVE_CHECKPOINT_DIR, options.SAVE_RESULTS_DIR_NAME, train=True)
    # save_config(config, run_id, options.SAVE_CHECKPOINT_DIR, options.SAVE_RESULTS_DIR_NAME, train=True)
    
    
    for epoch in range(options.NUM_EPOCHS):
    # for epoch in range(1):
        
        loop = tqdm(train_loader, leave=True)
        epoch_losses = []
        for index, images_labels in enumerate(loop):
            if dataset_version == 1:
                images = images_labels[0:3]
                img_id = images_labels[3]
                image_A, image_B, real_target_C, target_labels = get_data_for_input_mod(images, options.INPUT_MODALITIES)
                
            elif dataset_version == 2:           
                image_A, image_B, real_target_C, target_labels, img_id = images_labels[0], images_labels[1], images_labels[2], images_labels[3], images_labels[4]
                image_A, image_B, real_target_C, target_labels = reshape_data(image_A, image_B, real_target_C, target_labels)
                
            elif dataset_version == 3:           
                image_A, image_B, real_target_C, in_out_comb, img_id = images_labels[0], images_labels[1], images_labels[2], images_labels[3], images_labels[4]
                # import pdb; pdb.set_trace()
                image_A, image_B, real_target_C, in_out_comb = reshape_data(image_A, image_B, real_target_C, in_out_comb)
                in_out_comb = in_out_comb.to(options.DEVICE)
                target_labels = in_out_to_ohe_label(in_out_comb, 3)
                
            image_A, image_B, real_target_C, target_labels = image_A.to(options.DEVICE), image_B.to(options.DEVICE), real_target_C.to(options.DEVICE), target_labels.to(options.DEVICE)
            
            x_concat = torch.cat((image_A, image_B), dim=1)
            target_fake, image_A_recon, image_B_recon = gen(x_concat, target_labels)
            
            
            # backward of disc
            # -- Disc loss for fake --
            set_require_grad(disc, True)
            opt_disc.zero_grad()
            pred_disc_fake = disc(target_fake.detach(), target_labels) # as dont want to backward this 

            loss_D_fake = criterion_GAN(pred_disc_fake, torch.zeros_like(pred_disc_fake))
            
            # -- Disc loss for real --
            pred_disc_fake = disc(real_target_C, target_labels)
            loss_D_real = criterion_GAN(pred_disc_fake, torch.ones_like(pred_disc_fake))
            
            # get both loss and backprop
            loss_D = (loss_D_fake + loss_D_real) / 2
            loss_D.backward()
            opt_disc.step()
        
        
            # backward of gen
            set_require_grad(disc, False)
            opt_gen.zero_grad()
            
            pred_disc_fake = disc(target_fake, target_labels) # D(x)
            
            # loss for GAN
            loss_G_BCE = criterion_GAN(pred_disc_fake, torch.ones_like(pred_disc_fake))
            loss_G_L1 = criterion_L1(target_fake, real_target_C) * options.LAMBDA_GAN_L1
            
            # loss for reconstucting unet
            loss_G_reconA = criterion_L1(image_A_recon, image_A)
            loss_G_reconB = criterion_L1(image_B_recon, image_B)
            
            # loss for gradient difference between pred and real
            loss_G_GDL = criterion_GDL(target_fake, real_target_C)
            # options.LAMBDA_BCE = 10
            # options.LAMBDA_RECON = 1
            
            # loss_G = 20*loss_G_BCE + 100*loss_G_L1 + 20*loss_G_reconA + 20*loss_G_reconB 
            loss_G = options.LAMBDA_BCE*loss_G_BCE + loss_G_L1 + options.LAMBDA_RECON*loss_G_reconA + options.LAMBDA_RECON*loss_G_reconB + options.LAMBDA_GDL*loss_G_GDL
            loss_G.backward()
            opt_gen.step()

            loop.set_description(f"Epoch [{epoch+1}/{options.NUM_EPOCHS}]: Batch [{index+1}/{len(train_loader)}]")
            loop.set_postfix(loss_G=loss_G.item(), loss_D=loss_D.item())
            
            loss_dict = OrderedDict()
            
            batch_loss_dict = {
                "batch": index+1,
                "G_GAN": loss_G_BCE.item(),
                "G_L1": loss_G_L1.item(),
                "G_GDL": loss_G_GDL.item(),
                "D_real": loss_D_real.item(),
                "D_fake": loss_D_fake.item(),
            }
            epoch_losses.append(batch_loss_dict)
        
        scheduler_disc.step()
        scheduler_gen.step()
         
        log_loss_to_json(options.SAVE_CHECKPOINT_DIR, options.SAVE_RESULTS_DIR_NAME, run_id, epoch+1, epoch_losses)
        log_loss_to_txt(options.SAVE_CHECKPOINT_DIR, options.SAVE_RESULTS_DIR_NAME, run_id, epoch+1, loss_G_BCE, loss_G_L1, loss_G_reconA, loss_G_reconB, loss_G_GDL, loss_D_fake, loss_D_real)
           
        if options.SAVE_MODEL:
            if (epoch+1) > 5 and (epoch+1) % options.CHECKPOINT_INTERVAL == 0:
            # if (epoch+1) % options.CHECKPOINT_INTERVAL == 0:
                save_checkpoint(gen, opt_gen, checkpoint_dir=options.SAVE_CHECKPOINT_DIR, dir_name=options.SAVE_RESULTS_DIR_NAME, save_filename=f"{epoch+1}_net_G.pth")
                save_checkpoint(disc, opt_disc, checkpoint_dir=options.SAVE_CHECKPOINT_DIR, dir_name=options.SAVE_RESULTS_DIR_NAME, save_filename=f"{epoch+1}_net_D.pth")
                # print("checkpoint saved")
    
    end_time = time.time()  
    
    print(f"Time taken: {end_time - start_time:.4f} seconds")
            

    


        
