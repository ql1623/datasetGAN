import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

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

def print_model_structure(module, indent=0):
    print(' ' * indent + f'{module.__class__.__name__}')
    for name, child in module.named_children():
        print(' ' * (indent + 2) + f'{name}: {child.__class__.__name__}')
        print_model_structure(child, indent + 4)
        
        
if __name__ == "__main__":
    net_layer = config.NUM_LAYERS
    num_features = config.NUM_FEATURES
    print(f"Model architecture is {net_layer} layers, with {num_features} initial feature channels")
    
    if net_layer == 4:
        if config.CONDITION_METHOD == "concat":
            import model.cgan.generator_unet4_cgan as models
        elif config.CONDITION_METHOD == "add": # not concat but add
            import model.cgan.generator_unet4_cgan_v2 as models  
        else:  
            raise Exception("Condition method in GAN is not one of the predefined options")
    else:
        raise Exception("Number of UNET layers, net_layer is not specified to 4")
    
    num_modalities = 3   
    gen = models.datasetGAN(input_channels=1, output_channels=1, ngf=num_features).to(config.DEVICE)
    # summary(gen, (2, 128, 128))
    disc = models.Discriminator(in_channels=1, features=[32,64,128,256,512]).to(config.DEVICE)
    
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(config.B1, config.B2))  
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(config.B1, config.B2)) 
    
    epoch_to_load = 200
    gen, opt_gen = load_checkpoint(gen, opt_gen, config.LEARNING_RATE, config.SAVE_CHECKPOINT_DIR, config.SAVE_RESULTS_DIR_NAME, epoch_to_load, gen=True)
    disc, opt_disc = load_checkpoint(disc, opt_disc, config.LEARNING_RATE, config.SAVE_CHECKPOINT_DIR, config.SAVE_RESULTS_DIR_NAME, epoch_to_load, gen=False)
    
    test_dataset = MRI_dataset(config, transform=None, train=False, test=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    run_id = datetime.datetime.now().strftime("run_%d-%m-%Y_%H-%M-%S")
    save_config(config, run_id, config.SAVE_RESULTS_DIR, config.SAVE_RESULTS_DIR_NAME, train=False)
    
    for index, images_labels in enumerate(test_loader):
        
        images = images_labels[0:3]
        img_id = images_labels[3]
        image_A, image_B, real_target_C, target_labels = get_data_for_input_mod(images, config.INPUT_MODALITIES)
        
        image_A, image_B, real_target_C, target_labels = image_A.to(config.DEVICE), image_B.to(config.DEVICE), real_target_C.to(config.DEVICE), target_labels.to(config.DEVICE)
            
        x_concat = torch.cat((image_A, image_B), dim=1)
        
        with torch.no_grad():
            pred_target, pred_image_A_recon, pred_image_B_recon = gen(x_concat, target_labels)
            
            pred_images_C, real_images_C = patches_to_images(pred_target, real_target_C, [200,200], [2,2])
            real_images_A, real_images_B = patches_to_images(image_A, image_B, [200,200], [2,2])
            
            # import pdb; pdb.set_trace()
            avg_ssim, avg_psnr, avg_mse = evaluate_images(pred_images_C, real_images_C, run_id, index, config.SAVE_RESULTS_DIR, config.SAVE_RESULTS_DIR_NAME) 
            
            print("[" + f"Batch {index+1}: MSE: {avg_mse:.6f} | SSIM: {avg_ssim:.6f} | PSNR: {avg_psnr:.6f}" + "]")
        
        save_results(img_id, real_images_A, real_images_B, pred_images_C, real_images_C, config.SAVE_RESULTS_DIR, config.SAVE_RESULTS_DIR_NAME)
        # print("results was saved")
        
    generate_html(run_id, config.SAVE_RESULTS_DIR, config.SAVE_RESULTS_DIR_NAME)
    # print("html was generated")
    
    