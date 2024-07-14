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
    
#     gen = models.datasetGAN(input_channels=1, output_channels=1, ngf=ngf).to(config.DEVICE)
#     disc = models.Discriminator(in_channels=1, features=[32, 64, 128, 256, 512]).to(config.DEVICE)
    
#     B1 = trial.suggest_float('B1', 0.5, 0.9)
#     B2 = config.B2
#     LR = trial.suggest_float('LR', 2e-4, 1e-4)

#     opt_disc = optim.Adam(disc.parameters(), LR=LR, betas=(B1, B2))
#     opt_gen = optim.Adam(gen.parameters(), LR=LR, betas=(B1, B2))
    
#     return gen, disc, opt_gen, opt_disc


def objective(trial):
    # gen, disc, opt_gen, opt_disc = define_models(trial)
    # --- define generator and discriminator ---
    # model_name = trial.suggest_categorical('model_name', ['generator3', 'generator4'])
    # ngf = trial.suggest_categorical('ngf', [16, 32])
    model_name = "generator4"
    ngf = 32
    
    if model_name == 'generator3':
        import model.generator_unet3 as models
    elif model_name == 'generator4':
        import model.generator_unet4 as models
    
    gen = models.datasetGAN(input_channels=1, output_channels=1, ngf=ngf)
    disc = models.Discriminator(in_channels=1, features=[32, 64, 128, 256, 512])
    
    if torch.cuda.device_count() > 1:
        gen = nn.DataParallel(gen)
        disc = nn.DataParallel(disc)
    
    gen.to(config.DEVICE)
    disc.to(config.DEVICE)
    
    # apply_init = trial.suggest_categorical("apply_init", [True, False])
    apply_init = False
    if apply_init:
        gen.apply(initialize_weights)
        disc.apply(initialize_weights)
        
    # --- define optimizers ---
    # B1 = trial.suggest_categorical('B1', [0.5, 0.9])
    B1 = config.B1
    B2 = config.B2
    # LR = trial.suggest_categorical('LR', [1e-4, 2e-4])
    LR = config.LEARNING_RATE
    opt_disc = optim.Adam(disc.parameters(), lr=LR, betas=(B1, B2))
    opt_gen = optim.Adam(gen.parameters(), lr=LR, betas=(B1, B2))
    
    # --- define other hyperparams and lr scheduler ---
    # BATCH_SIZE = trial.suggest_categorical('BATCH_SIZE', [16, 32, 64])
    BATCH_SIZE = config.BATCH_SIZE
    # LR_START_EPOCH = trial.suggest_categorical('LR_START_EPOCH', [25, 50])
    LR_START_EPOCH = config.LR_START_EPOCH
    
    L1_LAMBDA = trial.suggest_categorical('L1_LAMBDA', [1.0, 10.0, 25.0, 50.0, 100.0])
    RECON_LAMBDA = 1.0
    BCE_LAMBDA = trial.suggest_categorical('BCE_LAMBDA', [1.0, 10.0, 25.0, 50.0])
    
    scheduler_gen = optim.lr_scheduler.LambdaLR(opt_gen, lr_lambda=lambda epoch: lambda_lr(epoch, LR_START_EPOCH, config.NUM_EPOCHS))
    scheduler_disc = optim.lr_scheduler.LambdaLR(opt_disc, lr_lambda=lambda epoch: lambda_lr(epoch, LR_START_EPOCH, config.NUM_EPOCHS))

    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()
    
    # --- define train and test dataset ---
    train_dataset = MRI_dataset(config, transform=None, train=True, test=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # run_id = datetime.datetime.now().strftime("run_%d-%m-%Y_%H-%M-%S")
    # save_config(config, run_id, config.SAVE_CHECKPOINT_DIR)
        
    for epoch in range(config.NUM_EPOCHS):
    # for epoch in range(1):
        loop = tqdm(train_loader, leave=True)
                
        for index, images in enumerate(train_loader):
            image_A, image_B, target_C = get_data_for_task(images, config.MODALITY_DIRECTION)
            image_A, image_B, target_C = image_A.to(config.DEVICE), image_B.to(config.DEVICE), target_C.to(config.DEVICE)
            
            x_concat = torch.cat((image_A, image_B), dim=1)
            target_fake, image_A_recon, image_B_recon = gen(x_concat)
            
            # Discriminator backward pass
            set_require_grad(disc, True)
            opt_disc.zero_grad()
            pred_disc_fake = disc(target_fake.detach())
            loss_D_fake = criterion_GAN(pred_disc_fake, torch.zeros_like(pred_disc_fake))
            pred_disc_real = disc(target_C)
            loss_D_real = criterion_GAN(pred_disc_real, torch.ones_like(pred_disc_real))
            loss_D = (loss_D_fake + loss_D_real) * 0.5
            loss_D.backward()
            opt_disc.step()
        
            # Generator backward pass
            set_require_grad(disc, False)
            opt_gen.zero_grad()
            pred_disc_fake = disc(target_fake)
            loss_G_BCE = criterion_GAN(pred_disc_fake, torch.ones_like(pred_disc_fake))
            loss_G_L1 = criterion_L1(target_fake, target_C) * L1_LAMBDA
            loss_G_reconA = criterion_L1(image_A_recon, image_A)
            loss_G_reconB = criterion_L1(image_B_recon, image_B)
            loss_G = BCE_LAMBDA * loss_G_BCE + loss_G_L1 + RECON_LAMBDA * loss_G_reconA + RECON_LAMBDA * loss_G_reconB
            loss_G.backward()
            opt_gen.step()

            loop.set_description(f"Epoch [{epoch + 1}/{config.NUM_EPOCHS}]: Batch [{index + 1}/{len(train_loader)}]")
            loop.set_postfix(loss_G=loss_G.item(), loss_D=loss_D.item())

        scheduler_disc.step()
        scheduler_gen.step()
        
        # Log the generator loss to Optuna
        trial.report(loss_G.item(), epoch)

        # Handle pruning (optional)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return loss_G.item()

def save_state(study, trial):
    with open("optuna/sampler_gpu4.pkl", "wb") as sample_file:
        pickle.dump(study.sampler, sample_file)
    with open("optuna/pruner_gpu4.pkl", "wb") as pruner_file:
        pickle.dump(study.pruner, pruner_file)
        
if __name__ == "__main__":
    # config = Config("./utils/params.yaml")
    # config = Options.parse()
    # print(config.BATCH_SIZE)
    storage_name = "sqlite:///optuna/optuna_study_gpu4.db"
    study_name = "GAN_optimization_gpu4"
    
    if os.path.exists("optuna/sampler_gpu4.pkl") and os.path.exists("optuna/pruner_gpu4.pkl"):
        with open("optuna/sampler_gpu4.pkl", "rb") as sample_file:
            restored_sampler = pickle.load(sample_file)
            print("previously saved SAMPLER was loaded")
        with open("optuna/pruner_gpu4.pkl", "rb") as pruner_file:
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
    
    