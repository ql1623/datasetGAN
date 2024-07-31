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

from utils.utils import *

import model.cgan.generator_unet4_cgan_v2_seg_v2 as models 
from dataset_png_v3_seg_v2 import MRI_dataset  

from train_options_v2 import TrainOptions

"""Have Segmentation Network separately and not inside GAN, take feature layers right before out_conv of fusion
generate 1 time only, 1 pass on each of 3 optimiser, 1st pass: disc opt, seg opt, gan opt
gan and seg = different optimiser, update disc -> update seg net -> update gan
removed other dataset version as will be using version 4 anyways""" 
        
if __name__ == "__main__":
    
    parser = TrainOptions()
    options = parser.parse()
    options.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # print(options.GPU_IDS)
     
    net_layer = options.NUM_LAYERS
    num_features = options.NUM_FEATURES
    seg_num_features = options.NUM_SEG_FEATURES
    print(f"Model architecture is {net_layer} layers, with gan = {num_features} feat, seg = {seg_num_features} feat")
    
    import model.cgan.generator_unet4_cgan_v2_seg_v2 as models 
    
    num_modalities = 3   
    gan_version = options.GAN_VERSION
    gen = models.datasetGAN(input_channels=1, output_channels=1, ngf=num_features, pre_out_channels=seg_num_features, version=gan_version)
    # summary(gen, (2, 128, 128))
    disc = models.Discriminator(in_channels=1, features=[32,64,128,256,512])
    seg = models.SegmentationNetwork(input_ngf=seg_num_features, output_channels=1)
    # seg = models.SegmentationNetwork(input_ngf=seg_num_features, output_channels=1)
    
    if torch.cuda.device_count() > 1:
        gen = nn.DataParallel(gen, options.GPU_IDS) # DistributedDataParallel?
        disc = nn.DataParallel(disc, options.GPU_IDS)
        seg = nn.DataParallel(seg, options.GPU_IDS)
    
    gen.to(options.DEVICE)
    disc.to(options.DEVICE)
    seg.to(options.DEVICE)
    
    opt_disc = optim.Adam(disc.parameters(), lr=options.LEARNING_RATE, betas=(options.B1, options.B2)) 
    opt_gen = optim.Adam(gen.parameters(), lr=options.LEARNING_RATE, betas=(options.B1, options.B2))  
    opt_seg = optim.Adam(seg.parameters(), lr=options.LEARNING_RATE, betas=(options.B1, options.B2))  
    
    epoch_to_load = options.LOAD_EPOCH
    gen, opt_gen = load_checkpoint(gen, opt_gen, options.LEARNING_RATE, options.SAVE_CHECKPOINT_DIR, options.LOAD_RESULTS_DIR_NAME, epoch_to_load, model_type="gen")
    disc, opt_disc = load_checkpoint(disc, opt_disc, options.LEARNING_RATE, options.SAVE_CHECKPOINT_DIR, options.LOAD_RESULTS_DIR_NAME, epoch_to_load, model_type="disc")
    seg, opt_seg = load_checkpoint(seg, opt_seg, options.LEARNING_RATE, options.SAVE_CHECKPOINT_DIR, options.LOAD_RESULTS_DIR_NAME, epoch_to_load, model_type="seg")
    
    dataset_version = options.DATASET_VERSION
    test_dataset = MRI_dataset(options.INPUT_MODALITIES, options.DATA_PNG_DIR, transform=True, train=False)
    test_loader = DataLoader(test_dataset, batch_size=options.BATCH_SIZE, shuffle=False, num_workers=options.NUM_WORKERS)
    
    run_id = datetime.datetime.now().strftime("run_%d-%m-%Y_%H-%M-%S")
    parser.save_options(run_id, options.SAVE_RESULTS_DIR, options.LOAD_RESULTS_DIR_NAME, train=False)
    # save_config(config, run_id, options.SAVE_RESULTS_DIR, options.LOAD_RESULTS_DIR_NAME, train=False)
    
    for index, images_labels in enumerate(test_loader):
        
        image_A, image_B, real_target_C, real_seg, in_out_comb, img_id = images_labels[0], images_labels[1], images_labels[2], images_labels[3], images_labels[4], images_labels[5]
        # import pdb; pdb.set_trace()
        in_out_comb = in_out_comb.to(options.DEVICE)
        target_labels = in_out_to_ohe_label(in_out_comb, 3)
        image_A, image_B, real_target_C, real_seg, target_labels = image_A.to(options.DEVICE), image_B.to(options.DEVICE), real_target_C.to(options.DEVICE), real_seg.to(options.DEVICE), target_labels.to(options.DEVICE)
            
        x_concat = torch.cat((image_A, image_B), dim=1)
        
        with torch.no_grad():
            # generate
            pred_target, pred_image_A_recon, pred_image_B_recon, fusion_features= gen(x_concat, target_labels)
            # # segment
            pred_seg = seg(fusion_features)
            # pred_images_C, real_images_C = patches_to_images(pred_target, real_target_C, [200,200], [2,2])
            # real_images_A, real_images_B = patches_to_images(image_A, image_B, [200,200], [2,2])
            # pred_seg, real_seg = patches_to_images(pred_seg, real_seg, [200,200], [2,2])
            
            # import pdb; pdb.set_trace()
            avg_ssim, avg_psnr, avg_mse, error_metrics = evaluate_images(pred_target, real_target_C, run_id, index, options.SAVE_RESULTS_DIR, options.LOAD_RESULTS_DIR_NAME) 
            
            print("[" + f"Batch {index+1}: MSE: {avg_mse:.6f} | SSIM: {avg_ssim:.6f} | PSNR: {avg_psnr:.6f}" + "]")
        
        # elif dataset_version == 2: 
        # save_results(img_id, target_labels, real_images_A, real_images_B, pred_target, real_target_C, options.SAVE_RESULTS_DIR, options.LOAD_RESULTS_DIR_NAME)
        # elif dataset_version == 3:  
        save_results_seg(run_id, img_id, in_out_comb, options.INPUT_MODALITIES, index, image_A, image_B, pred_target, real_target_C, pred_seg, real_seg, error_metrics, options.SAVE_RESULTS_DIR, options.LOAD_RESULTS_DIR_NAME)
        # print("results was saved")
        
    generate_html_seg(run_id, options.SAVE_RESULTS_DIR, options.LOAD_RESULTS_DIR_NAME)
    # print("html was generated")
    
    