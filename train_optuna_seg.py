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

import model.cgan.generator_unet4_cgan_v2_seg_v2 as models 
from dataset_png_v3_seg import MRI_dataset  

from train_options_v2 import TrainOptions

import optuna
from optuna.trial import TrialState
import sqlite3
import pickle

# def define_models(trial):
#     model_name = trial.suggest_categorical('model', ['generator3', 'generator4'])
    
#     if model_name == 'generator3':
#         import model.generator_syn_model_unet3 as models
#     elif model_name == 'generator4':
#         import model.generator_syn_model_unet4 as models

#     ngf = trial.suggest_int('ngf', 16, 32)
    
#     gen = models.datasetGAN(input_channels=1, output_channels=1, ngf=ngf).to(options.DEVICE)
#     disc = models.Discriminator(in_channels=1, features=[32, 64, 128, 256, 512]).to(options.DEVICE)
    
#     B1 = trial.suggest_float('B1', 0.5, 0.9)
#     B2 = options.B2
#     LR = trial.suggest_float('LR', 2e-4, 1e-4)

#     opt_disc = optim.Adam(disc.parameters(), LR=LR, betas=(B1, B2))
#     opt_gen = optim.Adam(gen.parameters(), LR=LR, betas=(B1, B2))
    
#     return gen, disc, opt_gen, opt_disc


def objective(trial):
    parser = TrainOptions()
    options = parser.parse()
    # gen, disc, opt_gen, opt_disc = define_models(trial)
    # --- define generator and discriminator ---
    # model_name = trial.suggest_categorical('model_name', ['generator3', 'generator4'])
    # ngf = trial.suggest_categorical('ngf', [16, 32])
    net_layer = options.NUM_LAYERS
    num_features = options.NUM_FEATURES
    seg_num_features = options.NUM_SEG_FEATURES
    
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
    
    # apply_init = trial.suggest_categorical("apply_init", [True, False])
    apply_init = False
    if apply_init:
        gen.apply(initialize_weights)
        disc.apply(initialize_weights)
        
    # --- define optimizers ---
    # B1 = trial.suggest_categorical('B1', [0.5, 0.9])
    B1 = options.B1
    B2 = options.B2
    # LR = trial.suggest_categorical('LR', [1e-4, 2e-4])
    LR = options.LEARNING_RATE
    opt_disc = optim.Adam(disc.parameters(), lr=LR, betas=(B1, B2)) 
    opt_gen = optim.Adam(gen.parameters(), lr=LR, betas=(B1, B2))  
    opt_seg = optim.Adam(seg.parameters(), lr=LR, betas=(B1, B2))  
    
    # --- define other hyperparams and lr scheduler ---
    # BATCH_SIZE = trial.suggest_categorical('BATCH_SIZE', [16, 32, 64])
    BATCH_SIZE = options.BATCH_SIZE
    # LR_START_EPOCH = trial.suggest_categorical('LR_START_EPOCH', [25, 50])
    LR_START_EPOCH = options.LR_START_EPOCH
    
    L1_LAMBDA = trial.suggest_categorical('L1_LAMBDA', [1.0, 10.0, 25.0])
    RECON_LAMBDA = 1.0
    GAN_BCE_LAMBDA = trial.suggest_categorical('GAN_BCE_LAMBDA', [1.0, 10.0, 25.0])
    LAMBDA_GDL = trial.suggest_categorical('LAMBDA_GDL', [0.1, 1.0])
    LAMBDA_SEG_BCE = trial.suggest_categorical('LAMBDA_SEG_BCE', [1.0, 10.0])
    LAMBDA_SEG_DICE = trial.suggest_categorical('LAMBDA_SEG_DICE', [0.1, 1.0, 10.0])
    
    scheduler_gen = optim.lr_scheduler.LambdaLR(opt_gen, lr_lambda=lambda epoch: lambda_lr(epoch, LR_START_EPOCH, options.NUM_EPOCHS))
    scheduler_disc = optim.lr_scheduler.LambdaLR(opt_disc, lr_lambda=lambda epoch: lambda_lr(epoch, LR_START_EPOCH, options.NUM_EPOCHS))
    scheduler_seg = optim.lr_scheduler.LambdaLR(opt_seg, lr_lambda=lambda epoch: lambda_lr(epoch, LR_START_EPOCH, options.NUM_EPOCHS))

    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()
    criterion_L1_GAN = L1LossWithAttention()
    # criterion_L2 = nn.MSELoss()
    criterion_GDL = GradientDifferenceLoss()
    criterion_SEG_BCE = nn.BCELoss()
    criterion_SEG_DICE = DiceLoss(epsilon=1e-6)
    
    # --- define train and test dataset ---
    # dataset_version = options.DATASET_VERSION
    train_dataset = MRI_dataset(options.INPUT_MODALITIES, options.DATA_PNG_DIR, transform=True, train=True) 
    train_loader = DataLoader(train_dataset, batch_size=options.BATCH_SIZE, shuffle=True, num_workers=options.NUM_WORKERS)
    
    # run_id = datetime.datetime.now().strftime("run_%d-%m-%Y_%H-%M-%S")
    # save_options(options, run_id, options.SAVE_CHECKPOINT_DIR)
        
    for epoch in range(options.NUM_EPOCHS):
    # for epoch in range(1):
        loop = tqdm(train_loader, leave=True)
                
        for index, images_labels in enumerate(train_loader):
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
            
            # Discriminator backward pass
            set_require_grad(disc, True)
            opt_disc.zero_grad()
            pred_disc_fake = disc(target_fake.detach(), target_labels) # as dont want to backward this 
            loss_D_fake = criterion_GAN(pred_disc_fake, torch.zeros_like(pred_disc_fake)) # D(G(x))
            pred_disc_real = disc(real_target_C, target_labels)
            loss_D_real = criterion_GAN(pred_disc_real, torch.ones_like(pred_disc_real)) # D(x)
            loss_D = (loss_D_fake + loss_D_real) / 2
            loss_D.backward()
            opt_disc.step()
        
            # Segmentation backward pass
            set_require_grad(disc, False)
            opt_seg.zero_grad()
            loss_S_BCE = criterion_SEG_BCE(seg_target_fake, real_seg) # S(G(x))
            loss_S_DICE = criterion_SEG_DICE(seg_target_fake, real_seg)
            loss_S = LAMBDA_SEG_BCE * loss_S_BCE + LAMBDA_SEG_DICE * loss_S_DICE
            loss_S.backward(retain_graph=True)
            opt_seg.step()           
            
            # Generator backward pass
            opt_gen.zero_grad()
            pred_disc_fake = disc(target_fake, target_labels) # D(G(x))
            loss_G_BCE = criterion_GAN(pred_disc_fake, torch.ones_like(pred_disc_fake))
            loss_G_L1 = criterion_L1_GAN(target_fake, real_target_C, seg_target_fake.detach()) 
            loss_G_reconA = criterion_L1(image_A_recon, image_A)
            loss_G_reconB = criterion_L1(image_B_recon, image_B)
            loss_G_GDL = criterion_GDL(target_fake, real_target_C)
            loss_G = GAN_BCE_LAMBDA * loss_G_BCE + L1_LAMBDA * loss_G_L1 + RECON_LAMBDA * loss_G_reconA + RECON_LAMBDA * loss_G_reconB + LAMBDA_GDL*loss_G_GDL
            loss_G_raw = loss_G_BCE + loss_G_L1 + loss_G_reconA + loss_G_reconB + loss_G_GDL
            loss_G.backward()
            opt_gen.step()

            loop.set_description(f"Epoch [{epoch + 1}/{options.NUM_EPOCHS}]: Batch [{index + 1}/{len(train_loader)}]")
            loop.set_postfix(loss_G=loss_G.item(), loss_D=loss_D.item(), loss_S=loss_S.item())

        scheduler_disc.step()
        scheduler_gen.step()
        scheduler_seg.step()
        
        # Log the generator loss to Optuna
        trial.report(loss_G_raw.item(), epoch)

        # Handle pruning (optional)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return loss_G_raw.item()

def save_state(study, trial):
    with open("optuna/sampler_seg_gpu.pkl", "wb") as sample_file:
        pickle.dump(study.sampler, sample_file)
    with open("optuna/pruner_seg_gpu.pkl", "wb") as pruner_file:
        pickle.dump(study.pruner, pruner_file)
        
if __name__ == "__main__":
    # options = options("./utils/params.yaml")
    # options = Options.parse()
    # print(options.BATCH_SIZE)
    storage_name = "sqlite:///optuna/optuna_study_seg_gpu.db"
    study_name = "GAN_optimization_seg_gpu"
    
    if os.path.exists("optuna/sampler_seg_gpu.pkl") and os.path.exists("optuna/pruner_seg_gpu.pkl"):
        with open("optuna/sampler_seg_gpu.pkl", "rb") as sample_file:
            restored_sampler = pickle.load(sample_file)
            print("previously saved SAMPLER was loaded")
        with open("optuna/pruner_seg_gpu.pkl", "rb") as pruner_file:
            restored_pruner = pickle.load(pruner_file)
            print("previously saved PRUNER was loaded")
            
        study = optuna.create_study(
            study_name=study_name, storage=storage_name, load_if_exists=True,
            sampler=restored_sampler, pruner=restored_pruner
        )
    else:
        study = optuna.create_study(direction="minimize", storage=storage_name, study_name=study_name)

    # set total optimization time limit to 70 hours just to be sure
    try:
        # study.optimize(objective, n_trials=100, timeout=252000, callbacks=[lambda study, trial: save_state(study, storage_name, study_name)])
        study.optimize(objective, n_trials=100, timeout=252000, callbacks=[save_state])
    finally:
        # Make sure state is saved at the end of the optimization
        save_state(study, None)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print(f"Value: {trial.value}")

    print("Hyperparameters:")
    for key, value in trial.params.items():
        print(f"{key}: {value}")
    
    # # Create a study object, using SQLite as the backend
    # study = optuna.create_study(direction="minimize", storage=storage_name, study_name="GAN_optimization", load_if_exists=True)
    
    # # Set the total optimization time limit to 72 hours (259200 seconds)
    # study.optimize(objective, n_trials=1000, timeout=259200)

    # print("Best trial:")
    # trial = study.best_trial

    # print(f"Value: {trial.value}")

    # print("Hyperparameters:")
    # for key, value in trial.params.items():
    #     print(f"{key}: {value}")
    
    