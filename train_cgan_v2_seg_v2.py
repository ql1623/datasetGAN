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

import matplotlib.pyplot as plt
from utils.config_reader import Config
# from train_options import Options
# import train_options as config
from utils.utils import *

from train_options_v2 import TrainOptions

"""Have Segmentation Network separately and not inside GAN"""

def save_probability_map(seg_target_fake, checkpoint_dir, dir_name, filename="probability_map.npy"):
    # Apply sigmoid to get probabilities
    prob_map = torch.sigmoid(seg_target_fake).detach().cpu().numpy()

    # Assuming batch size is 1 for simplicity
    # prob_map = prob_map[0, 0, :, :]  # Get the first sample in the batch and squeeze out the channel dimension

    # Save the probability map as a .npy file
    save_path = os.path.join(checkpoint_dir, "chkpt_" + dir_name, filename)
    np.save(save_path, prob_map)
    print(f"Probability map saved to {save_path}")


def plot_probability_map(seg_target_fake):
    # Apply sigmoid to get probabilities
    prob_map = torch.sigmoid(seg_target_fake).detach().cpu().numpy()

    # Assuming batch size is 1 for simplicity
    prob_map = prob_map[0, 0, :, :]  # Get the first sample in the batch and squeeze out the channel dimension

    # Plot the probability map
    plt.imshow(prob_map, cmap='gray')
    plt.colorbar()
    plt.title('Probability Map')
    plt.show()

def patches_to_images2(pred_patches, image_ori_shape, num_patches):
    h, w = image_ori_shape # [160, 200]
    num_patches_h, num_patches_w = num_patches # [2,2]
    batch_with_patch_size, _, patch_size_h, patch_size_w = pred_patches.shape
    batch_size = batch_with_patch_size // (num_patches_h * num_patches_w) # floor (//) or int(), as division (/) gets float 
    stride_h = (h-patch_size_h) // (num_patches_h - 1)
    stride_w = (w-patch_size_w) // (num_patches_w - 1)
    
    batch_pred_patches = torch.reshape(pred_patches, (batch_size, 4, pred_patches.shape[2], pred_patches.shape[3])) # [batch_size, 4, 128, 128]
    # batch_real_patches = torch.reshape(real_patches, (batch_size, 4, real_patches.shape[2], real_patches.shape[3]))

    # Iterate over every possible position of the kernel
    recon_pred_images = torch.zeros((batch_size, h, w), dtype=pred_patches.dtype)
    # recon_real_images = torch.zeros((batch_size, h, w), dtype=real_patches.dtype)
    overlap_count = torch.zeros((batch_size, h, w), dtype=torch.float32)
    
    for batch_index in range(batch_size):
        patch_idx = 0
        for i in range(0, h - patch_size_h + 1, stride_h):
            for j in range(0, w - patch_size_w + 1, stride_w):
                recon_pred_images[batch_index, i:i + patch_size_h, j:j + patch_size_w] += batch_pred_patches[batch_index, patch_idx].cpu()
                # recon_real_images[batch_index, i:i + patch_size_h, j:j + patch_size_w] += batch_real_patches[batch_index, patch_idx].cpu()
                overlap_count[batch_index, i:i + patch_size_h, j:j + patch_size_w] += 1
                patch_idx += 1
    
    pred_images = recon_pred_images / overlap_count
    # real_images = recon_real_images / overlap_count
    
    for batch_index in range(batch_size):
        pred_images[batch_index, :, :] = normalize_image(pred_images[batch_index])
        # real_images[batch_index, :, :] = normalize_image(real_images[batch_index])
        
    pred_images = torch.unsqueeze(pred_images, 1)
    # real_images = torch.unsqueeze(real_images, 1)
        
    return pred_images #, real_images

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
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
            # import model.cgan.generator_unet4_cgan_v2_seg as models  
            import model.cgan.generator_unet4_cgan_v2_seg_v2 as models  
        else:  
            raise Exception("Condition method in GAN is not one of the predefined options")
    else:
        raise Exception("Number of UNET layers, net_layer is not specified to 4")
        
    num_modalities = 3   
    gan_version = options.GAN_VERSION
    gen = models.datasetGAN(input_channels=1, output_channels=1, ngf=num_features, version=gan_version)
    # summary(gen, (2, 128, 128))
    disc = models.Discriminator(in_channels=1, features=[32,64,128,256,512])
    seg = models.SegmentationNetwork(input_ngf=num_features, output_channels=1)
    
    if torch.cuda.device_count() > 1:
        gen = nn.DataParallel(gen, options.GPU_IDS) # DistributedDataParallel?
        disc = nn.DataParallel(disc, options.GPU_IDS)
        seg = nn.DataParallel(seg, options.GPU_IDS)
    
    gen.to(options.DEVICE)
    disc.to(options.DEVICE)
    seg.to(options.DEVICE)
        
    # initialize weights inside
    # gen.apply(initialize_weights)
    # disc.apply(initialize_weights)
    
    opt_disc = optim.Adam(disc.parameters(), lr=options.LEARNING_RATE, betas=(options.B1, options.B2)) 
    opt_gen = optim.Adam(gen.parameters(), lr=options.LEARNING_RATE, betas=(options.B1, options.B2))  
    opt_seg = optim.Adam(seg.parameters(), lr=options.LEARNING_RATE, betas=(options.B1, options.B2))  
    
    # learning rate decay
    scheduler_gen = optim.lr_scheduler.LambdaLR(opt_gen, lr_lambda=lambda epoch: lambda_lr(epoch, options.LR_START_EPOCH, options.NUM_EPOCHS))
    scheduler_disc = optim.lr_scheduler.LambdaLR(opt_disc, lr_lambda=lambda epoch: lambda_lr(epoch, options.LR_START_EPOCH, options.NUM_EPOCHS))
    scheduler_seg = optim.lr_scheduler.LambdaLR(opt_seg, lr_lambda=lambda epoch: lambda_lr(epoch, options.LR_START_EPOCH, options.NUM_EPOCHS))

    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()
    criterion_L1_GAN = L1LossWithAttention()
    # criterion_L2 = nn.MSELoss()
    criterion_GDL = GradientDifferenceLoss()
    criterion_SEG = nn.BCELoss()
    
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
    elif dataset_version == 4:
        from dataset_png_v3_seg import MRI_dataset  
        train_dataset = MRI_dataset(options.INPUT_MODALITIES, options.DATA_PNG_DIR, transform=True, train=True) 
    train_loader = DataLoader(train_dataset, batch_size=options.BATCH_SIZE, shuffle=True, num_workers=options.NUM_WORKERS)
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=options.NUM_WORKERS)
    
    run_id = datetime.datetime.now().strftime("run_%H:%M:%S_%d/%m/%Y")
    parser.save_options(run_id, options.SAVE_CHECKPOINT_DIR, options.SAVE_RESULTS_DIR_NAME, train=True)
    # save_config(config, run_id, options.SAVE_CHECKPOINT_DIR, options.SAVE_RESULTS_DIR_NAME, train=True)
    
    # for epoch in range(options.NUM_EPOCHS):
    for epoch in range(1):
        
        loop = tqdm(train_loader, leave=True)
        epoch_losses = []
        for index, images_labels in enumerate(loop):
            if dataset_version == 1:
                images = images_labels[0:3]
                img_id = images_labels[3]
                image_A, image_B, real_target_C, target_labels = get_data_for_input_mod(images, options.INPUT_MODALITIES)
                image_A, image_B, real_target_C, target_labels = image_A.to(options.DEVICE), image_B.to(options.DEVICE), real_target_C.to(options.DEVICE), target_labels.to(options.DEVICE)
            
            elif dataset_version == 2:           
                image_A, image_B, real_target_C, target_labels, img_id = images_labels[0], images_labels[1], images_labels[2], images_labels[3], images_labels[4]
                image_A, image_B, real_target_C, target_labels = reshape_data(image_A, image_B, real_target_C, target_labels)
                image_A, image_B, real_target_C, target_labels = image_A.to(options.DEVICE), image_B.to(options.DEVICE), real_target_C.to(options.DEVICE), target_labels.to(options.DEVICE)
                
            elif dataset_version == 3:           
                image_A, image_B, real_target_C, in_out_comb, img_id = images_labels[0], images_labels[1], images_labels[2], images_labels[3], images_labels[4]
                # import pdb; pdb.set_trace()
                image_A, image_B, real_target_C, in_out_comb = reshape_data(image_A, image_B, real_target_C, in_out_comb)
                in_out_comb = in_out_comb.to(options.DEVICE)
                target_labels = in_out_to_ohe_label(in_out_comb, 3)
                image_A, image_B, real_target_C, target_labels = image_A.to(options.DEVICE), image_B.to(options.DEVICE), real_target_C.to(options.DEVICE), target_labels.to(options.DEVICE)
                    
                    
            elif dataset_version == 4:        
                image_A, image_B, real_target_C, real_seg, in_out_comb, img_id = images_labels[0], images_labels[1], images_labels[2], images_labels[3], images_labels[4], images_labels[5]
                # import pdb; pdb.set_trace()
                image_A, image_B, real_target_C, real_seg, in_out_comb = reshape_data_seg(image_A, image_B, real_target_C, real_seg, in_out_comb)
                in_out_comb = in_out_comb.to(options.DEVICE)
                target_labels = in_out_to_ohe_label(in_out_comb, 3)
                image_A, image_B, real_target_C, real_seg, target_labels = image_A.to(options.DEVICE), image_B.to(options.DEVICE), real_target_C.to(options.DEVICE), real_seg.to(options.DEVICE), target_labels.to(options.DEVICE)
                
            x_concat = torch.cat((image_A, image_B), dim=1)
            # generate
            target_fake, image_A_recon, image_B_recon, fusion_features = gen(x_concat, target_labels)
            # segment
            seg_target_fake = seg(fusion_features)
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

            loss_D_fake = criterion_GAN(pred_disc_fake, torch.zeros_like(pred_disc_fake)) # D(G(x))
            
            # -- Disc loss for real --
            pred_disc_real = disc(real_target_C, target_labels)
            loss_D_real = criterion_GAN(pred_disc_real, torch.ones_like(pred_disc_real)) # D(x)
            
            # get both loss and backprop
            loss_D = (loss_D_fake + loss_D_real) / 2
            loss_D.backward()
            opt_disc.step()
            # print("disc")
            
            # ----- backward of seg ----- 
            # loss for segmentation
            set_require_grad(disc, False)
            opt_seg.zero_grad()
            loss_S = criterion_SEG(seg_target_fake, real_seg) * options.LAMBDA_SEG # S(G(x))
            loss_S.backward(retain_graph=True)
            opt_seg.step()
            # print("seg")
            
            # ----- backward of gen ----- 
            opt_gen.zero_grad()
            
            pred_disc_fake = disc(target_fake, target_labels) # D(G(x))
            
            # loss for GAN
            loss_G_BCE = criterion_GAN(pred_disc_fake, torch.ones_like(pred_disc_fake))
            # loss_G_L1 = criterion_L1(target_fake, real_target_C) 
            loss_G_L1 = criterion_L1_GAN(target_fake, real_target_C, seg_target_fake.detach()) 
            
            # loss for reconstucting unet
            loss_G_reconA = criterion_L1(image_A_recon, image_A)
            loss_G_reconB = criterion_L1(image_B_recon, image_B)
            
            # loss for gradient difference between pred and real
            loss_G_GDL = criterion_GDL(target_fake, real_target_C)
                        
            # options.LAMBDA_BCE = 10
            # options.LAMBDA_RECON = 1
            
            # loss_G = 20*loss_G_BCE + 100*loss_G_L1 + 20*loss_G_reconA + 20*loss_G_reconB 
            loss_G = options.LAMBDA_BCE*loss_G_BCE + options.LAMBDA_GAN_L1*loss_G_L1 + options.LAMBDA_RECON*loss_G_reconA + options.LAMBDA_RECON*loss_G_reconB + options.LAMBDA_GDL*loss_G_GDL
            loss_G.backward()
            opt_gen.step()
            # print("gen")

            loop.set_description(f"Epoch [{epoch+1}/{options.NUM_EPOCHS}]: Batch [{index+1}/{len(train_loader)}]")
            loop.set_postfix(loss_G=loss_G.item(), loss_D=loss_D.item(), loss_S=loss_S.item())
            
            # loss_dict = OrderedDict()
            
            batch_loss_dict = {
                "batch": index+1,
                "G_GAN": loss_G_BCE.item(),
                "G_L1": loss_G_L1.item(),
                "G_GDL": loss_G_GDL.item(),
                "D_real": loss_D_real.item(),
                "D_fake": loss_D_fake.item(),
                "S_fake": loss_S.item(),
            }
            epoch_losses.append(batch_loss_dict)
            
        scheduler_disc.step()
        scheduler_gen.step()
  
        log_loss_to_json(options.SAVE_CHECKPOINT_DIR, options.SAVE_RESULTS_DIR_NAME, run_id, epoch+1, epoch_losses)
        log_loss_to_txt(options.SAVE_CHECKPOINT_DIR, options.SAVE_RESULTS_DIR_NAME, run_id, epoch+1, 
                        loss_data=[loss_G_BCE, loss_G_L1, loss_G_reconA, loss_G_reconB, loss_G_GDL, loss_D_fake, loss_D_real, loss_S], 
                        loss_name=["G_BCE", "G_L1", "G_reA", "G_reB", "G_GDL", "D_fake", "D_real", "S_fake"])

        if options.SAVE_MODEL:
            if (epoch+1) > 5 and (epoch+1) % options.CHECKPOINT_INTERVAL == 0:
            # if (epoch+1) % options.CHECKPOINT_INTERVAL == 0:
                save_checkpoint(gen, opt_gen, checkpoint_dir=options.SAVE_CHECKPOINT_DIR, dir_name=options.SAVE_RESULTS_DIR_NAME, save_filename=f"{epoch+1}_net_G.pth")
                save_checkpoint(disc, opt_disc, checkpoint_dir=options.SAVE_CHECKPOINT_DIR, dir_name=options.SAVE_RESULTS_DIR_NAME, save_filename=f"{epoch+1}_net_D.pth")
            # if (epoch+1) > 10 and (epoch+1) % options.CHECKPOINT_INTERVAL == 0:
                save_checkpoint(seg, opt_seg, checkpoint_dir=options.SAVE_CHECKPOINT_DIR, dir_name=options.SAVE_RESULTS_DIR_NAME, save_filename=f"{epoch+1}_net_S.pth")
                # print("checkpoint saved")
    
    end_time = time.time()  
    
    print(f"Time taken: {end_time - start_time:.4f} seconds")
            

    


        
