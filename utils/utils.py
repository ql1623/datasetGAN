import torch
import torch.nn.init as init
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import os
import json

import random
from itertools import permutations
# import train_options as config

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse

# def mkdirs(paths):
#     """create empty directories if they don't exist

#     Parameters:
#         paths (str list) -- a list of directory paths
#     """
#     if isinstance(paths, list) and not isinstance(paths, str):
#         for path in paths:
#             mkdir(path)
#     else:
#         mkdir(paths)
        
def mkdir(path):
    """create an empty directory if it didn't exist

    Parameters:
        path (str): a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
        
        
def set_require_grad(model, require_grad=False):
    """ to set if model's weights are to be updated / frozen

    Parameters:
        model (nn.Module): the model to set whether gradient calculation is needed
        require_grad (bool): if gradient is needed
    """
    for param in model.parameters():
        param.requires_grad = require_grad


def initialize_weights(m):
    """Initialize the weights and biases in the layers of the neural network"""
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find("Conv") != -1):
        init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
    elif hasattr(m, 'children'):
        for _, child in m.named_children():
            initialize_weights(child)
    
        
def lambda_lr(epoch, lr_start_epoch, num_epochs):    
    if epoch < lr_start_epoch:
            return 1.0
    else:
        return max(0.0, 1.0 - float(epoch - lr_start_epoch) / float(num_epochs - lr_start_epoch))


def save_config(config, run_id, file_dir, dir_name, train=True):
    if train:
        save_path = os.path.join(file_dir, "chkpt_" + dir_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        with open(os.path.join(save_path, "train_opt.txt"), 'a') as file:
            file.write(f"================ Training Options {run_id} ================\n")
            for key, value in config.__dict__.items():
                if not key.startswith('__') and not callable(value) and not isinstance(value, type(config)):
                    file.write(f"{key}: {value}\n")
                    
    else:
        save_path = os.path.join(file_dir, dir_name + "test")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        with open(os.path.join(save_path, "test_opt.txt"), 'a') as file:
            file.write(f"================ Testing Options {run_id} ================\n")
            for key, value in config.__dict__.items():
                if not key.startswith('__') and not callable(value) and not isinstance(value, type(config)):
                    file.write(f"{key}: {value}\n")
        
             
def log_loss_to_txt(log_dir, dir_name, run_id, epoch, loss_G_BCE, loss_G_L1, loss_G_reconA, loss_G_reconB, loss_D_fake, loss_D_real):
    save_path = os.path.join(log_dir, "chkpt_" + dir_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    log_file_path = os.path.join(save_path, "loss_log.txt")
    
    with open(log_file_path, 'a') as file:
        if epoch == 1:
            # Write header if the it is start of training
            file.write(f"================ Training Loss {run_id} ================\n")
            file.write(f"epoch:\tG_BCE\t\tG_L1\t\tG_reA\t\tG_reB\t\tD_fake\t\tD_real\n")
        file.write(f"{epoch}\t\t{loss_G_BCE:.6f}\t{loss_G_L1:.6f}\t{loss_G_reconA:.6f}\t{loss_G_reconB:.6f}\t{loss_D_fake:.6f}\t{loss_D_real:.6f}\n")
        file.flush() 

def log_loss_to_json(log_dir, dir_name, run_id, epoch, losses):
    save_path = os.path.join(log_dir, "chkpt_" + dir_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    log_file_path = os.path.join(save_path, "loss_log.json")

    if os.path.exists(log_file_path):
        with open(log_file_path, 'r') as file:
            logs = json.load(file)
    else:
        logs = {}

    if run_id not in logs:
        logs[run_id] = []

    log_entry = {
        "epoch": epoch,
        "losses": losses
    }
    logs[run_id].append(log_entry)

    with open(log_file_path, 'w') as file:
        json.dump(logs, file, indent=4)
        

def save_checkpoint(model, optimizer, checkpoint_dir, dir_name, save_filename):
    save_dir = os.path.join(checkpoint_dir, "chkpt_" + dir_name)
    save_path = os.path.join(save_dir, save_filename)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }    
    torch.save(checkpoint, save_path)

def load_checkpoint(model, optimizer, lr, checkpoint_dir, dir_name, epoch_num, gen=True):
    save_dir = os.path.join(checkpoint_dir, "chkpt_" + dir_name)
    if gen == True:
        save_path = os.path.join(save_dir, f"{epoch_num}_net_G.pth")
    else:
        save_path = os.path.join(save_dir, f"{epoch_num}_net_D.pth")
    print(save_path)
    if not os.path.exists(save_path):
        print("No such checkpoint is in this directory")
         
    # checkpoint = torch.load(save_path, map_location=config.DEVICE)
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    
    return model, optimizer

def get_patches(image, patch_size, num_patches):

    h, w = image.shape
    patch_size_h, patch_size_w = patch_size
    num_patches_h, num_patches_w = num_patches
    stride_h = (h-patch_size_h) // (num_patches_h - 1)
    stride_w = (w-patch_size_w) // (num_patches_w - 1)
    # overlap_h = patch_size-(h-patch_size)
    # overlap_w = patch_size-(w-patch_size)
    patches = np.reshape(np.lib.stride_tricks.sliding_window_view(image, patch_size)[::stride_h, ::stride_w], (num_patches_h*num_patches_w, patch_size_h, patch_size_w))
    
    return patches
    
    
# def reconstruct_2d_image(patches, np_data_shape, stride):
#     h, w = np_data_shape
    
def normalize_image(image):
    """Normalize the image to the range [-1, 1]."""
    min_val = image.min()
    max_val = image.max()
    normalized_image = 2 * (image - min_val) / (max_val - min_val) - 1
    return normalized_image.float()

    
def patches_to_images(pred_patches, real_patches, image_ori_shape, num_patches):
    h, w = image_ori_shape # [160, 200]
    num_patches_h, num_patches_w = num_patches # [2,2]
    batch_with_patch_size, _, patch_size_h, patch_size_w = pred_patches.shape
    batch_size = batch_with_patch_size // (num_patches_h * num_patches_w) # floor (//) or int(), as division (/) gets float 
    stride_h = (h-patch_size_h) // (num_patches_h - 1)
    stride_w = (w-patch_size_w) // (num_patches_w - 1)
    
    batch_pred_patches = torch.reshape(pred_patches, (batch_size, 4, pred_patches.shape[2], pred_patches.shape[3])) # [batch_size, 4, 128, 128]
    batch_real_patches = torch.reshape(real_patches, (batch_size, 4, real_patches.shape[2], real_patches.shape[3]))

    # Iterate over every possible position of the kernel
    recon_pred_images = torch.zeros((batch_size, h, w), dtype=pred_patches.dtype)
    recon_real_images = torch.zeros((batch_size, h, w), dtype=real_patches.dtype)
    overlap_count = torch.zeros((batch_size, h, w), dtype=torch.float32)
    
    for batch_index in range(batch_size):
        patch_idx = 0
        for i in range(0, h - patch_size_h + 1, stride_h):
            for j in range(0, w - patch_size_w + 1, stride_w):
                recon_pred_images[batch_index, i:i + patch_size_h, j:j + patch_size_w] += batch_pred_patches[batch_index, patch_idx].cpu()
                recon_real_images[batch_index, i:i + patch_size_h, j:j + patch_size_w] += batch_real_patches[batch_index, patch_idx].cpu()
                overlap_count[batch_index, i:i + patch_size_h, j:j + patch_size_w] += 1
                patch_idx += 1
    
    pred_images = recon_pred_images / overlap_count
    real_images = recon_real_images / overlap_count
    
    for batch_index in range(batch_size):
        pred_images[batch_index, :, :] = normalize_image(pred_images[batch_index])
        real_images[batch_index, :, :] = normalize_image(real_images[batch_index])
        
    pred_images = torch.unsqueeze(pred_images, 1)
    real_images = torch.unsqueeze(real_images, 1)
        
    return pred_images, real_images


def evaluate_images(pred_images, real_images, run_id, batch_index, eval_log_dir, dir_name): # pred_images and real_images [32,1,160,200]
    
    batch_size = pred_images.shape[0]
    error_metrics= {
        "SSIM": [],
        "PSNR": [],
        "MSE": [],
    }
    # epsilon = 1e-6
    for i in range(batch_size):
        pred_image = torch.squeeze(pred_images[i]).cpu().numpy()
        real_image = torch.squeeze(real_images[i]).cpu().numpy()
    
        error_metrics["SSIM"].append(ssim(real_image, pred_image, data_range=pred_image.max() - pred_image.min()))
        error_metrics["PSNR"].append(psnr(real_image, pred_image, data_range=pred_image.max() - pred_image.min()))
        error_metrics["MSE"].append(mse(real_image, pred_image))
    
    avg_ssim = np.mean(error_metrics["SSIM"])
    avg_psnr = np.mean(error_metrics["PSNR"])
    avg_mse = np.mean(error_metrics["MSE"])
    
    if eval_log_dir:
        if dir_name:
            eval_log_path = os.path.join(eval_log_dir, dir_name)
        else:
            eval_log_path = eval_log_dir
        if not os.path.exists(eval_log_path):
            os.makedirs(eval_log_path)
            
        eval_log_file = os.path.join(eval_log_path, "test_results.txt")
        
        with open(eval_log_file, 'a') as log_file:
            if batch_index == 0:
                log_file.write(f"================ Testing Model {run_id} ================\n")
            log_file.write("[" + f"Batch {batch_index+1}: MSE: {avg_mse:.6f} | SSIM: {avg_ssim:.6f} | PSNR: {avg_psnr:.6f}" + "]\n")

    return avg_ssim, avg_psnr, avg_mse


# def save_results(image_A, image_B, pred_images, real_images):
    
def normalize_image_for_png(image):
    """
    Normalize the slice data to the range 0-255 and convert to uint8.
    """
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val - min_val  > 0:  # Avoid division by zero
        normalized_image  = (image - min_val ) / (max_val - min_val ) * 255
    else:
        normalized_image  = np.zeros_like(image)  # Avoid division by zero
    return normalized_image.astype(np.uint8)

def save_image(tensor, png_save_path):
    """Save a tensor as an image."""
    # image = tensor.cpu().numpy().transpose(1, 2, 0)  # convert from [C, H, W] --> [H, W, C]
    image = tensor.cpu().numpy().squeeze(0)  # convert from [C, H, W] --> [H, W, C]
    # Scale to [0, 255] and convert to uint8 in order to store as png
    image = normalize_image_for_png(image)
    # cv2.imwrite(path, image.squeeze())
    png_image = Image.fromarray(image)
    png_image.save(png_save_path)


def save_results(batch_id, image_A, image_B, pred_images, real_images, save_results_dir, dir_name):
    """Save images to a specified folder."""
    save_path = os.path.join(save_results_dir, dir_name, "images")
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    for i in range(image_A.shape[0]):
        id_str = f"Batch_{batch_id+1}_{i+1}"
        
        save_image(image_A[i], os.path.join(save_path, f"{id_str}_real_A.png"))
        save_image(image_B[i], os.path.join(save_path, f"{id_str}_real_B.png"))
        save_image(pred_images[i], os.path.join(save_path, f"{id_str}_fake_C.png"))
        save_image(real_images[i], os.path.join(save_path, f"{id_str}_real_C.png"))


def generate_html(run_id, save_results_dir, dir_name):
    """Generate an HTML file to view the images."""
    save_images_path = os.path.join(save_results_dir, dir_name, "images")
    save_html_path = os.path.join(save_results_dir, dir_name)
    html_content = f"<html><body><h2>Testing Results {dir_name}</h2>"
    pred_images = {}
    
    for filename in sorted(os.listdir(save_images_path)):
        if filename.endswith(".png"):
            batch_id = "_".join(filename.split("_")[:3])
            # print(batch_id)
            if batch_id not in pred_images:
                pred_images[batch_id] = []
            pred_images[batch_id].append(filename)
    
    # print(pred_images.keys())        
    for batch_id, files in pred_images.items():
        html_content += f'<h3>{batch_id}</h3>'
        html_content += '<div style="display: flex;">'
        img_sequence = ['real_A', 'real_B', 'fake_C', 'real_C']
        for file in files:
            for type in img_sequence:
                if type in file:
                    html_content += f'<div style="text-align: center; margin: 10px;">'
                    html_content += f'<img src="images/{file}" style="max-width: 200px; display: block;"><br><p>{file}</p>'
                    html_content += '</div>'
        html_content += '</div><br>'
    
    html_content += "</body></html>"
    
    with open(os.path.join(save_html_path, f"{run_id}.html"), "w") as f:
        f.write(html_content)




def get_data_for_task(images, modality_direction):
    # images from dataloader are in: [t1_slice, t1ce_slice, t2_slice, flair_slice], each with size [batch_size, 4, 128, 128]
    if modality_direction == "t1_t1ce_to_t2" or "t1ce_t1_to_t2":
        image_A = torch.reshape(images[0],(images[0].shape[0] * images[0].shape[1], 1, images[0].shape[2], images[0].shape[3])).type(torch.FloatTensor) 
        image_B = torch.reshape(images[1],(images[1].shape[0] * images[1].shape[1], 1, images[1].shape[2], images[1].shape[3])).type(torch.FloatTensor) 
        image_C = torch.reshape(images[2],(images[2].shape[0] * images[2].shape[1], 1, images[2].shape[2], images[2].shape[3])).type(torch.FloatTensor) 
        
    elif modality_direction == "t1_t1ce_to_flair" or "t1ce_t1_to_flair":
        image_A = torch.reshape(images[0],(images[0].shape[0] * images[0].shape[1], 1, images[0].shape[2], images[0].shape[3])).type(torch.FloatTensor) 
        image_B = torch.reshape(images[1],(images[1].shape[0] * images[1].shape[1], 1, images[1].shape[2], images[1].shape[3])).type(torch.FloatTensor) 
        image_C = torch.reshape(images[3],(images[3].shape[0] * images[3].shape[1], 1, images[3].shape[2], images[3].shape[3])).type(torch.FloatTensor) 
        
    elif modality_direction == "t1ce_t2_to_t1" or "t2_t1ce_to_t1":
        image_A = torch.reshape(images[1],(images[1].shape[0] * images[1].shape[1], 1, images[1].shape[2], images[1].shape[3])).type(torch.FloatTensor) 
        image_B = torch.reshape(images[2],(images[2].shape[0] * images[2].shape[1], 1, images[2].shape[2], images[2].shape[3])).type(torch.FloatTensor) 
        image_C = torch.reshape(images[0],(images[0].shape[0] * images[0].shape[1], 1, images[0].shape[2], images[0].shape[3])).type(torch.FloatTensor) 
        
    elif modality_direction == "t1ce_t2_to_flair" or "t2_t1ce_to_flair":
        image_A = torch.reshape(images[1],(images[1].shape[0] * images[1].shape[1], 1, images[1].shape[2], images[1].shape[3])).type(torch.FloatTensor) 
        image_B = torch.reshape(images[2],(images[2].shape[0] * images[2].shape[1], 1, images[2].shape[2], images[2].shape[3])).type(torch.FloatTensor) 
        image_C = torch.reshape(images[3],(images[3].shape[0] * images[3].shape[1], 1, images[3].shape[2], images[3].shape[3])).type(torch.FloatTensor) 
        
    elif modality_direction == "t2_flair_to_t1" or "flair_t2_to_t1":
        image_A = torch.reshape(images[2],(images[2].shape[0] * images[2].shape[1], 1, images[2].shape[2], images[2].shape[3])).type(torch.FloatTensor) 
        image_B = torch.reshape(images[3],(images[3].shape[0] * images[3].shape[1], 1, images[3].shape[2], images[3].shape[3])).type(torch.FloatTensor) 
        image_C = torch.reshape(images[0],(images[0].shape[0] * images[0].shape[1], 1, images[0].shape[2], images[0].shape[3])).type(torch.FloatTensor) 
        
    elif modality_direction == "t2_flair_to_t1ce" or "flair_t2_to_t1ce":
        image_A = torch.reshape(images[2],(images[2].shape[0] * images[2].shape[1], 1, images[2].shape[2], images[2].shape[3])).type(torch.FloatTensor) 
        image_B = torch.reshape(images[3],(images[3].shape[0] * images[3].shape[1], 1, images[3].shape[2], images[3].shape[3])).type(torch.FloatTensor) 
        image_C = torch.reshape(images[1],(images[1].shape[0] * images[1].shape[1], 1, images[1].shape[2], images[1].shape[3])).type(torch.FloatTensor) 
    
    elif modality_direction == "t1_t2_to_t1ce" or "t2_t1_to_t1ce":
        image_A = torch.reshape(images[0],(images[0].shape[0] * images[0].shape[1], 1, images[0].shape[2], images[0].shape[3])).type(torch.FloatTensor) 
        image_B = torch.reshape(images[2],(images[2].shape[0] * images[2].shape[1], 1, images[2].shape[2], images[2].shape[3])).type(torch.FloatTensor) 
        image_C = torch.reshape(images[1],(images[1].shape[0] * images[1].shape[1], 1, images[1].shape[2], images[1].shape[3])).type(torch.FloatTensor) 
        
    elif modality_direction == "t1_t2_to_flair" or "t2_t1_to_flair":
        image_A = torch.reshape(images[0],(images[0].shape[0] * images[0].shape[1], 1, images[0].shape[2], images[0].shape[3])).type(torch.FloatTensor) 
        image_B = torch.reshape(images[2],(images[2].shape[0] * images[2].shape[1], 1, images[2].shape[2], images[2].shape[3])).type(torch.FloatTensor) 
        image_C = torch.reshape(images[3],(images[3].shape[0] * images[3].shape[1], 1, images[3].shape[2], images[3].shape[3])).type(torch.FloatTensor) 
        
    elif modality_direction == "t1_flair_to_t1ce" or "flair_t1_to_t1ce":
        image_A = torch.reshape(images[0],(images[0].shape[0] * images[0].shape[1], 1, images[0].shape[2], images[0].shape[3])).type(torch.FloatTensor) 
        image_B = torch.reshape(images[3],(images[3].shape[0] * images[3].shape[1], 1, images[3].shape[2], images[3].shape[3])).type(torch.FloatTensor) 
        image_C = torch.reshape(images[1],(images[1].shape[0] * images[1].shape[1], 1, images[1].shape[2], images[1].shape[3])).type(torch.FloatTensor) 
        
    elif modality_direction == "t1_flair_to_t2" or "flair_t1_to_t2":
        image_A = torch.reshape(images[0],(images[0].shape[0] * images[0].shape[1], 1, images[0].shape[2], images[0].shape[3])).type(torch.FloatTensor) 
        image_B = torch.reshape(images[3],(images[3].shape[0] * images[3].shape[1], 1, images[3].shape[2], images[3].shape[3])).type(torch.FloatTensor) 
        image_C = torch.reshape(images[2],(images[2].shape[0] * images[2].shape[1], 1, images[2].shape[2], images[2].shape[3])).type(torch.FloatTensor) 
        
    elif modality_direction == "t1ce_flair_to_t2" or "flair_t1ce_to_t2":
        image_A = torch.reshape(images[1],(images[1].shape[0] * images[1].shape[1], 1, images[1].shape[2], images[1].shape[3])).type(torch.FloatTensor) 
        image_B = torch.reshape(images[3],(images[3].shape[0] * images[3].shape[1], 1, images[3].shape[2], images[3].shape[3])).type(torch.FloatTensor) 
        image_C = torch.reshape(images[2],(images[2].shape[0] * images[2].shape[1], 1, images[2].shape[2], images[2].shape[3])).type(torch.FloatTensor) 
        
    elif modality_direction == "t1ce_flair_to_t1" or "flair_t1ce_to_t1":
        image_A = torch.reshape(images[1],(images[1].shape[0] * images[1].shape[1], 1, images[1].shape[2], images[1].shape[3])).type(torch.FloatTensor) 
        image_B = torch.reshape(images[3],(images[3].shape[0] * images[3].shape[1], 1, images[3].shape[2], images[3].shape[3])).type(torch.FloatTensor) 
        image_C = torch.reshape(images[0],(images[0].shape[0] * images[0].shape[1], 1, images[0].shape[2], images[0].shape[3])).type(torch.FloatTensor) 
        
    return image_A, image_B, image_C


def check_input_seq(input_seq, valid_triplets):
    input_term = input_seq.split("_")
    
    if len(input_term) == 3: # something like "t1_t1ce_t2"
        input_set = set(input_seq.split('_'))
        for triplet in valid_triplets:
            if input_set == set(triplet.split('_')):
                return True
        return False

    if len(input_term) == 4: # something like "t1ce_t1_to_flair"
        pass
        # first_mod = input_term[0]
        # second_mod = input_term[1]
        
    else: 
        print("Input Sequence / Modality Direction specified is in wrong format")
         

def get_data_for_input_mod(images, input_modalities):
    # images from dataloader are in: [t1_slice, t1ce_slice, t2_slice, flair_slice], each with size [batch_size, 4, 128, 128]
    valid_triplets = [
        "t1_t1ce_t2",
        "t1_t1ce_flair",
        "t1_t2_flair",
        "t1ce_t2_flair"
    ]
    
    if check_input_seq(input_modalities, valid_triplets):
        # modalities_map = {
        #     't1': 0,
        #     't1ce': 1,
        #     't2': 2,
        #     'flair': 3
        # }
        modalities_map = {
            't1': 0,
            't2': 1,
            'flair': 2
        }
        modalities = input_modalities.split("_") # something like "t1_t1ce_t2"
        image_mod_num = [modalities_map[mod] for mod in modalities]
            
        in_out_combinations = [
            (image_mod_num[0], image_mod_num[1], image_mod_num[2]),
            (image_mod_num[0], image_mod_num[2], image_mod_num[1]),
            (image_mod_num[1], image_mod_num[2], image_mod_num[0])
        ]
        
        batch_size = images[0].shape[0]
        image_A_list, image_B_list, image_C_list, target_labels_list = [], [], [], []
        for i in range(batch_size):
            img_A_mod_num, img_B_mod_num, img_C_mod_num = random.choice(in_out_combinations)
            image_A_list.append(images[img_A_mod_num][i])
            image_B_list.append(images[img_B_mod_num][i])
            image_C_list.append(images[img_C_mod_num][i])
            target_labels_list.append(torch.tensor([img_C_mod_num] * images[img_C_mod_num].shape[1]))
            
        images_A_batch = torch.unsqueeze(torch.cat(image_A_list, dim=0), dim=1).type(torch.FloatTensor)
        images_B_batch = torch.unsqueeze(torch.cat(image_B_list, dim=0), dim=1).type(torch.FloatTensor)
        images_C_batch = torch.unsqueeze(torch.cat(image_C_list, dim=0), dim=1).type(torch.FloatTensor)
        target_labels_batch = get_ohe_label_vec(torch.cat(target_labels_list, dim=0), len(modalities_map), images_C_batch.shape[2])
        # print(target_labels_batch.shape)
    else:
        # raise ValueError("Invalid input modalities format")
        print("Invalid input modalities format")
    
    return images_A_batch, images_B_batch, images_C_batch, target_labels_batch

def get_ohe_label_vec(output_modality_num, num_classes, img_size):
    ohe_labels = F.one_hot(output_modality_num, num_classes).unsqueeze(2).unsqueeze(3)
    return ohe_labels.repeat(1, 1, img_size, img_size)

# def get_condition_label(modality_direction, num_classes):
#     output_mod = modality_direction.split("_")[-1]
#     out_mod_ohe_label = get_ohe_label(output_mod, num_classes)
    
#     return out_mod_ohe_label

if __name__ == "__main__":
    # modality_direction = "t1_t2_to_flair"
    # num_classes = 3
    # condition_modality = get_condition_label(modality_direction, num_classes)
    # print(condition_modality)
    # print(condition_modality.shape)
    
    
    def check_input_seq(input_seq, valid_triplets):
        input_term = input_seq.split("_")
        
        if len(input_term) == 3: # something like "t1_t1ce_t2"
            input_set = set(input_seq.split('_'))
            for triplet in valid_triplets:
                if input_set == set(triplet.split('_')):
                    return True
            return False

        if len(input_term) == 4: # something like "t1ce_t1_to_flair"
            pass
            # first_mod = input_term[0]
            # second_mod = input_term[1]
            
        else: 
            print("Input Sequence / Modality Direction specified is in wrong format")\
                
    valid_triplets = [
        "t1_t1ce_t2",
        "t1_t1ce_flair",
        "t1_t2_flair",
        "t1ce_t2_flair"
    ]
    # "t1_t1ce_t2" or "t1_t2_t1ce" or "t1ce_t2_t1"  or "t1ce_t1_t2" or "t2_t1_t1ce" or "t2_t1ce_t1"
    
    # "t1_t1ce_t2" or "t1_t1ce_flair" or "t1ce_t2_t1" or "t1ce_t2_flair" or "t2_flair_t1" or "t2_flair_t1ce" or 
    # "t1_t2_t1ce" or "t1_t2_flair" or "t1_flair_t1ce" or "t1_flair_t2" or "t1ce_flair_t2" or "t1ce_flair_t1"

    # print(check_input_seq("t1ce_flair_t1", valid_triplets))
    input_modalities = "t1_t1ce_flair"
    modalities_map = {
            't1': 0,
            't1ce': 1,
            't2': 2,
            'flair': 3
        }
    modalities = input_modalities.split("_") # something like "t1_t1ce_t2"
    img_A_mod_num = modalities_map[modalities[0]]
    img_B_mod_num = modalities_map[modalities[1]]
    img_C_mod_num = modalities_map[modalities[2]]
    
    print(img_A_mod_num)
    print(img_B_mod_num)
    print(img_C_mod_num)

    