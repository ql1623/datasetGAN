import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler, SubsetRandomSampler
from torchsummary import summary

from sklearn.model_selection import KFold
import numpy as np
import scipy.io as scio
import time
import datetime
from tqdm import tqdm
from torch.autograd import Variable
import torch.autograd as autograd

from collections import OrderedDict

import wandb

import matplotlib.pyplot as plt
from utils.utils import *

import model.cgan.generator_unet4_cgan_v2_seg_v3 as models 
from dataset_png_v3_seg_v2 import MRI_dataset  

from train_options_v2 import TrainOptions

"""Have Segmentation Network together and inside GAN, same architecture as GAN to see
   generate 1 time only, training with 1 pass on each of 2 optimiser, 1st pass: disc opt & (gan and seg) opt
   gan and seg = same optimiser, update disc -> update (gan and seg net) together 
   removed other dataset version as will be using version 4 anyways""" 

if __name__ == "__main__":
    wandb.login()
    
    torch.autograd.set_detect_anomaly(True)
    start_time = time.time()
    
    parser = TrainOptions()
    options = parser.parse()
    wandb.init(project="t1_t2_flair_gan", name=options.SAVE_RESULTS_DIR_NAME, config=options)

    options.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(options.GPU_IDS)
     
    net_layer = options.NUM_LAYERS
    num_features = options.NUM_FEATURES
    seg_num_features = options.NUM_SEG_FEATURES
    print(f"Model architecture is {net_layer} layers, with gan = {num_features} feat, seg = {seg_num_features} feat")
    
    import model.cgan.generator_unet4_cgan_v2_seg_v3 as models 
        
    # num_modalities = 3   
    gan_version = options.GAN_VERSION
    gen = models.datasetGAN(input_channels=1, output_channels=1, ngf=num_features, version=gan_version)
    # gen = models.datasetGAN(input_channels=1, output_channels=1, ngf=num_features, pre_out_channels=seg_num_features, version=gan_version)
    # summary(gen, (2, 128, 128))
    disc = models.Discriminator(in_channels=1, features=[32,64,128,256,512])
    # seg = models.SegmentationNetwork(input_ngf=seg_num_features, output_channels=1)
    # seg = models.SegmentationNetwork(input_ngf=seg_num_features, output_channels=1)
    
    if torch.cuda.device_count() > 1:
        gen = nn.DataParallel(gen, options.GPU_IDS) # DistributedDataParallel?
        disc = nn.DataParallel(disc, options.GPU_IDS)
        # seg = nn.DataParallel(seg, options.GPU_IDS)
    
    gen.to(options.DEVICE)
    disc.to(options.DEVICE)
    # seg.to(options.DEVICE)
        
    # initialize weights inside
    # gen.apply(initialize_weights)
    # disc.apply(initialize_weights)
    
    opt_disc = optim.Adam(disc.parameters(), lr=options.LEARNING_RATE, betas=(options.B1, options.B2)) 
    opt_gen = optim.Adam(gen.parameters(), lr=options.LEARNING_RATE, betas=(options.B1, options.B2))  
    # opt_seg = optim.Adam(seg.parameters(), lr=options.LEARNING_RATE, betas=(options.B1, options.B2))  
    
    # learning rate decay
    scheduler_gen = optim.lr_scheduler.LambdaLR(opt_gen, lr_lambda=lambda epoch: lambda_lr(epoch, options.LR_START_EPOCH, options.NUM_EPOCHS))
    scheduler_disc = optim.lr_scheduler.LambdaLR(opt_disc, lr_lambda=lambda epoch: lambda_lr(epoch, options.LR_START_EPOCH, options.NUM_EPOCHS))
    # scheduler_seg = optim.lr_scheduler.LambdaLR(opt_seg, lr_lambda=lambda epoch: lambda_lr(epoch, options.LR_START_EPOCH, options.NUM_EPOCHS))

    criterion_GAN_BCE = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()
    criterion_GAN_L1 = nn.L1Loss()
    criterion_GAN_L1_ATT = L1LossWithAttention()
    criterion_GAN_L2 = nn.MSELoss()
    criterion_GAN_L2_ATT = L2LossWithAttention()
    filter_type = options.GDL_TYPE
    if ("sobel" in filter_type) or ("prewitt" in filter_type):
        criterion_GDL = GradientDifferenceLoss(filter_type=options.GDL_TYPE)
    elif "canny" in filter_type:
        criterion_GDL = GradientDifferenceLossCanny(use_cuda=torch.cuda.is_available())
    else:
        raise Exception("filter type defined with wrong format")
    criterion_SEG_BCE = nn.BCELoss()
    criterion_SEG_DICE = DiceLoss(epsilon=1e-6)
    
    dataset_version = options.DATASET_VERSION
    # from dataset_png_v3_seg_v2 import MRI_dataset  
    train_dataset = MRI_dataset(options.INPUT_MODALITIES, options.DATA_PNG_DIR, transform=True, train=True) 
    train_loader = DataLoader(train_dataset, batch_size=options.BATCH_SIZE, shuffle=True, num_workers=options.NUM_WORKERS)
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=options.NUM_WORKERS)
    
    run_id = datetime.datetime.now().strftime("run_%H:%M:%S_%d/%m/%Y")
    parser.save_options(run_id, options.SAVE_CHECKPOINT_DIR, options.SAVE_RESULTS_DIR_NAME, train=True)
    # save_config(config, run_id, options.SAVE_CHECKPOINT_DIR, options.SAVE_RESULTS_DIR_NAME, train=True)
    
    for epoch in range(options.NUM_EPOCHS):
    # for epoch in range(1):
        
        loop = tqdm(train_loader, leave=True)
        epoch_losses = []
        for index, images_labels in enumerate(loop):
              
            image_A, image_B, real_target_C, real_seg, in_out_comb, img_id = images_labels[0], images_labels[1], images_labels[2], images_labels[3], images_labels[4], images_labels[5]
            # import pdb; pdb.set_trace()
            in_out_comb = in_out_comb.to(options.DEVICE)
            target_labels = in_out_to_ohe_label(in_out_comb, 3)
            image_A, image_B, real_target_C, real_seg, target_labels = image_A.to(options.DEVICE), image_B.to(options.DEVICE), real_target_C.to(options.DEVICE), real_seg.to(options.DEVICE), target_labels.to(options.DEVICE)
                
            x_concat = torch.cat((image_A, image_B), dim=1)
            # generate
            target_fake, image_A_recon, image_B_recon, seg_target_fake = gen(x_concat, target_labels)
            # segment
            # seg_target_fake = seg(fusion_features)
            # print(seg_target_fake.shape)
            # print(real_seg.shape)
            # import pdb; pdb.set_trace()
            # seg_target_fake_whole = patches_to_images2(seg_target_fake, [200,200], [2,2])
        
            # seg_target_fake_whole = seg_target_fake_whole.to('cpu')  # Move to CPU if it's on GPU
            # # plot_probability_map(seg_target_fake)
            # save_probability_map(seg_target_fake_whole, options.SAVE_CHECKPOINT_DIR, options.SAVE_RESULTS_DIR_NAME, filename="probability_map.npy")
            # # reveresed_confidence_map = 1 - seg_target_fake
            
            # ----- backward of disc ----- 
            # -- Disc loss for fake --
            set_require_grad(disc, True)
            opt_disc.zero_grad()
            pred_disc_fake = disc(target_fake.detach(), target_labels) # as dont want to backward this 

            loss_D_fake = criterion_GAN_BCE(pred_disc_fake, torch.zeros_like(pred_disc_fake)) # D(G(x))
            
            # -- Disc loss for real --
            pred_disc_real = disc(real_target_C, target_labels)
            loss_D_real = criterion_GAN_BCE(pred_disc_real, torch.ones_like(pred_disc_real)) # D(x)
            
            # get both loss and backprop
            loss_D = (loss_D_fake + loss_D_real) / 2
            loss_D.backward()
            opt_disc.step()
            # print("disc")
            
            # ----- backward of seg that is in the GAN now ----- 
            # loss for segmentation
            set_require_grad(disc, False)
            # opt_seg.zero_grad()
            # loss_S_BCE = criterion_SEG_BCE(seg_target_fake, real_seg) # S(G(x))
            # loss_S_DICE = criterion_SEG_DICE(seg_target_fake, real_seg)
            # loss_S = options.LAMBDA_SEG_BCE * loss_S_BCE + options.LAMBDA_SEG_DICE * loss_S_DICE
            # loss_S.backward(retain_graph=True)
            # opt_seg.step()
            # print("seg")
            
            # ----- backward of gen ----- 
            opt_gen.zero_grad()
            
            pred_disc_fake = disc(target_fake, target_labels) # D(G(x))
            
            # loss for GAN
            loss_G_BCE = criterion_GAN_BCE(pred_disc_fake, torch.ones_like(pred_disc_fake))
            loss_G_L1 = criterion_GAN_L1(target_fake, real_target_C) 
            
            # if needed
            loss_G_L2 = criterion_GAN_L2(target_fake, real_target_C) if options.USE_GAN_L2 else None
            loss_G_L1_ATT = criterion_GAN_L1_ATT(target_fake, real_target_C, seg_target_fake.detach()) if options.USE_GAN_L1_ATT else None
            loss_G_L2_ATT = criterion_GAN_L2_ATT(target_fake, real_target_C, seg_target_fake.detach()) if options.USE_GAN_L2_ATT else None
              
            # loss for reconstucting unet
            loss_G_reconA = criterion_L1(image_A_recon, image_A)
            loss_G_reconB = criterion_L1(image_B_recon, image_B)
            
            # loss for gradient difference between pred and real
            loss_G_GDL = criterion_GDL(target_fake, real_target_C)
            
            # loss for segmentation decoding branch
            loss_S_BCE = criterion_SEG_BCE(seg_target_fake, real_seg) # S(G(x))
            loss_S_DICE = criterion_SEG_DICE(seg_target_fake, real_seg)
            loss_S = options.LAMBDA_SEG_BCE * loss_S_BCE + options.LAMBDA_SEG_DICE * loss_S_DICE
            
            # loss_G = 20*loss_G_BCE + 100*loss_G_L1 + 20*loss_G_reconA + 20*loss_G_reconB 
            loss_G = (loss_S + 
                      options.LAMBDA_GAN_BCE * loss_G_BCE + 
                      options.LAMBDA_GAN_L1 * loss_G_L1 + 
                      options.LAMBDA_RECON_A * loss_G_reconA + 
                      options.LAMBDA_RECON_B * loss_G_reconB + 
                      options.LAMBDA_GDL * loss_G_GDL)
            
            if loss_G_L2 is not None:
                loss_G += options.LAMBDA_GAN_L2 * loss_G_L2
            if loss_G_L1_ATT is not None:
                loss_G += options.LAMBDA_GAN_L1_ATT * loss_G_L1_ATT
            if loss_G_L2_ATT is not None:
                loss_G += options.LAMBDA_GAN_L2_ATT * loss_G_L2_ATT
            loss_G.backward()
            opt_gen.step()
            # print("gen")

            loop.set_description(f"Epoch [{epoch+1}/{options.NUM_EPOCHS}]: Batch [{index+1}/{len(train_loader)}]")
            # loop.set_postfix(loss_G=loss_G.item(), loss_D=loss_D.item(), loss_S=loss_S.item())
            loop.set_postfix(loss_G=loss_G.item(), loss_D=loss_D.item())

            # loss_dict = OrderedDict()
            
            batch_loss_dict = {
                "batch": index+1,
                "G_GAN": loss_G_BCE.item(),
                "G_L1": loss_G_L1.item(),
                "G_reA": loss_G_reconA.item(),
                "G_reB": loss_G_reconB.item(),
                "G_GDL": loss_G_GDL.item(),
                "D_real": loss_D_real.item(),
                "D_fake": loss_D_fake.item(),
                "S_BCE": loss_S_BCE.item(),
                "S_DICE": loss_S_DICE.item(),
            }
            if loss_G_L2 is not None:
                batch_loss_dict["G_L2"] = loss_G_L2.item()
            if loss_G_L1_ATT is not None:
                batch_loss_dict["G_L1_ATT"] = loss_G_L1_ATT.item()
            if loss_G_L2_ATT is not None:
                batch_loss_dict["G_L2_ATT"] = loss_G_L2_ATT.item()
                
            epoch_losses.append(batch_loss_dict)
            
            log_data = {k: v for k, v in batch_loss_dict.items() if k != "batch"}
            wandb.log(log_data, step=epoch + 1)

        scheduler_disc.step()
        scheduler_gen.step()
        # scheduler_gen.step()
  
        log_loss_data = [loss_G_BCE, loss_G_L1, loss_G_reconA, loss_G_reconB, loss_G_GDL, loss_D_fake, loss_D_real, loss_S_BCE, loss_S_DICE]
        log_loss_names = ["G_BCE", "G_L1", "G_reA", "G_reB", "G_GDL", "D_fake", "D_real", "S_BCE", "S_DICE"]
        if loss_G_L2 is not None:
            log_loss_data.append(loss_G_L2)
            log_loss_names.append("G_L2")
        if loss_G_L1_ATT is not None:
            log_loss_data.append(loss_G_L1_ATT)
            log_loss_names.append("G_L1_ATT")
        if loss_G_L2_ATT is not None:
            log_loss_data.append(loss_G_L2_ATT)
            log_loss_names.append("G_L2_ATT")

        log_loss_to_json(options.SAVE_CHECKPOINT_DIR, options.SAVE_RESULTS_DIR_NAME, run_id, epoch+1, epoch_losses)
        log_loss_to_txt(options.SAVE_CHECKPOINT_DIR, options.SAVE_RESULTS_DIR_NAME, run_id, epoch+1, 
                        loss_data=log_loss_data, 
                        loss_name=log_loss_names)

        if options.SAVE_MODEL:
            if (epoch+1) > 5 and (epoch+1) % options.CHECKPOINT_INTERVAL == 0:
            # if (epoch+1) % options.CHECKPOINT_INTERVAL == 0:
                save_checkpoint(gen, opt_gen, checkpoint_dir=options.SAVE_CHECKPOINT_DIR, dir_name=options.SAVE_RESULTS_DIR_NAME, save_filename=f"{epoch+1}_net_G.pth")
                save_checkpoint(disc, opt_disc, checkpoint_dir=options.SAVE_CHECKPOINT_DIR, dir_name=options.SAVE_RESULTS_DIR_NAME, save_filename=f"{epoch+1}_net_D.pth")
            # if (epoch+1) > 10 and (epoch+1) % options.CHECKPOINT_INTERVAL == 0:
                # save_checkpoint(seg, opt_seg, checkpoint_dir=options.SAVE_CHECKPOINT_DIR, dir_name=options.SAVE_RESULTS_DIR_NAME, save_filename=f"{epoch+1}_net_S.pth")
                # print("checkpoint saved")

        # Log images to wandb at the end of each epoch
        target_fake_np = target_fake.detach().cpu().numpy()
        real_target_C_np = real_target_C.detach().cpu().numpy()
        seg_target_fake_np = seg_target_fake.detach().cpu().numpy()
        real_seg_np = real_seg.detach().cpu().numpy()
        
        # Log the first image of the batch
        wandb.log({
            "Target Fake": [wandb.Image(target_fake_np[0], caption="Target Fake")],
            "Real Target C": [wandb.Image(real_target_C_np[0], caption="Real Target C")],
            "Seg Target Fake": [wandb.Image(seg_target_fake_np[0], caption="Seg Target Fake")],
            "Real Seg": [wandb.Image(real_seg_np[0], caption="Real Seg")]
        })
        
    end_time = time.time()  
    
    print(f"Time taken: {end_time - start_time:.4f} seconds")
            

    


        
