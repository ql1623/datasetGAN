import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from PIL import Image
import random
import itertools
import time

from utils.utils import *
import train_options as config

# change target labels to in_out_combination chosen instead of one hot from v2

class BalancedChooser:
    def __init__(self, combinations):
        self.combinations = combinations
        self.reset()
    
    def reset(self):
        random.shuffle(self.combinations)
        self.iterator = itertools.cycle(self.combinations)
    
    def choose(self):
        if hasattr(self, 'remains') and self.remains == 0:
            self.reset()
        if not hasattr(self, 'remains'): 
            self.remains = len(self.combinations) - 1
        else: 
            self.remains -= 1
        return next(self.iterator)

class BalancedRandomChoice:
    def __init__(self, items):
        self.items = items
        self.reset()
    
    def reset(self):
        self.choices = self.items[:]
        random.shuffle(self.choices)
    
    def choose(self):
        if not self.choices:
            self.reset()
        return self.choices.pop()
        
        
        
class MRI_dataset(Dataset):
    def __init__(self, input_modalities, data_png_dir, transform=True, train=True):
        # self.config = config
        self.input_modalities = input_modalities
        self.data_png_dir = data_png_dir
        self.train = train
        
        modalities = self.input_modalities.split("_")
        mod_A, mod_B, mod_C = modalities[0], modalities[1], modalities[2]

        self.img_paths = {
            0: os.path.join(self.data_png_dir, mod_A, 'train' if train else 'test'),
            1: os.path.join(self.data_png_dir, mod_B, 'train' if train else 'test'),
            2: os.path.join(self.data_png_dir, mod_C, 'train' if train else 'test')
        }
        self.img_lists = {
            'A': os.listdir(self.img_paths[0]),
            'B': os.listdir(self.img_paths[1]),
            'C': os.listdir(self.img_paths[2])
        }
        
        if transform:
            self.transform = transforms.Compose([
                transforms.ToTensor(), # already normalize the image to [0, 1]
                transforms.Normalize(mean=[0.5], std=[0.5]) # then normalize the image to [-1, 1]
            ])        
        
        self.valid_triplets = [
            "t1_t1ce_t2",
            "t1_t1ce_flair",
            "t1_t2_flair",
            "t1ce_t2_flair"
        ]
        self.in_out_combinations = [
            (0, 1, 2),
            (0, 2, 1),
            (1, 2, 0)
        ]
        self.random_choice = BalancedChooser(self.in_out_combinations)
        # self.random_choice = BalancedRandomChoice(self.in_out_combinations)

    def check_input_seq(self, input_seq):
        input_term = input_seq.split("_")
        
        if len(input_term) == 3: # something like "t1_t1ce_t2"
            input_set = set(input_seq.split('_'))
            for triplet in self.valid_triplets:
                if input_set == set(triplet.split('_')):
                    return True
            return False

        elif len(input_term) == 4: # something like "t1ce_t1_to_flair"
            pass
        else: 
            print("Input Sequence / Modality Direction specified is in wrong format")
            return False
        
        return False

    def get_key(self, value, mod_dict):
        for k, v in mod_dict.items():
            if value == v:
                return k
        return "Corresponding key not found for this value"


    def get_patches(self, image, patch_size, num_patches):
        image = image.squeeze(0)
        h, w = image.shape
        patch_size_h, patch_size_w = patch_size
        num_patches_h, num_patches_w = num_patches
        stride_h = (h-patch_size_h) // (num_patches_h - 1)
        stride_w = (w-patch_size_w) // (num_patches_w - 1)
        patches = []
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                patch_index_h = i * stride_h
                patch_index_w = j * stride_w
                patch = image[patch_index_h:patch_index_h + patch_size_h, patch_index_w:patch_index_w + patch_size_w]
                patches.append(patch)
        
        return torch.stack(patches)

    def get_ohe_label_vec(self, output_modality_num, num_classes):
        return F.one_hot(output_modality_num, num_classes)

    def __getitem__(self, idx):
        patch_size = [128, 128]
        num_patches = [2, 2]
        
        img_id = self.img_lists["A"][idx].split(".")[0]
        
        img_A_ori_idx, img_B_ori_idx, img_C_ori_idx = self.random_choice.choose()
        
        pil_A = Image.open(os.path.join(self.img_paths[img_A_ori_idx], self.img_lists["A"][idx]))
        pil_B = Image.open(os.path.join(self.img_paths[img_B_ori_idx], self.img_lists["B"][idx]))
        pil_C = Image.open(os.path.join(self.img_paths[img_C_ori_idx], self.img_lists["C"][idx]))
        tensor_A = self.transform(pil_A)
        tensor_B = self.transform(pil_B)
        tensor_C = self.transform(pil_C)
        
        image_A = self.get_patches(tensor_A, patch_size, num_patches)
        image_B = self.get_patches(tensor_B, patch_size, num_patches)
        image_C = self.get_patches(tensor_C, patch_size, num_patches)
        
        if not self.check_input_seq(self.input_modalities):
            print("Invalid input modalities format")
            return None, None, None, None
     
        # target_labels = get_ohe_label_vec(torch.tensor([img_C_ori_idx] * 4), 3)
        in_out_comb = torch.tensor([img_A_ori_idx, img_B_ori_idx, img_C_ori_idx]).view(1,3).repeat(4, 1)

        return image_A, image_B, image_C, in_out_comb, img_id
    
    def __len__(self): 
        return len(self.img_lists["A"])

if __name__ == "__main__":
    print(config.BATCH_SIZE)
    print(config.INPUT_MODALITIES)

    start_time = time.time()

    train_data = MRI_dataset(config.INPUT_MODALITIES, config.DATA_PNG_DIR, transform=True, train=True)
    print(len(train_data))
    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)

    for index, images_labels in enumerate(train_loader):
        image_A, image_B, target_C, in_out_comb, img_id = images_labels[0], images_labels[1], images_labels[2], images_labels[3], images_labels[4]
        print(in_out_comb.shape)
        print(len(img_id))
        print(type(img_id))
        print("---------")
        
        print(image_A[0].min())
        print(image_A[0].max())
        print("---------")
        
        image_A, image_B, target_C, in_out_comb = reshape_data(image_A, image_B, target_C, in_out_comb)
        print(in_out_comb.shape)
        break
    
    c = 0
    for i, id in enumerate(img_id):
        if c < 5:
            print(i,": " + img_id[0])
            c+=1
    
    print("---------")
    print(in_out_comb[0])
    print(in_out_comb[1])
    print(in_out_comb[2])
    print(in_out_comb[3])
    print(in_out_comb[4])
    print(in_out_comb[5])
    print(in_out_comb[6])
    print(in_out_comb[7])
    print(in_out_comb[8])
    print(in_out_comb[112])
    print(in_out_comb[45])
    print(in_out_comb[32])
    end_time = time.time()  
    
    

    print(f"Time taken: {end_time - start_time:.4f} seconds")
    
# Model architecture is 4 layers, with 32 intiial feature channels
# dataset_png, dataset_version = 1
# num_workers = 2
# Time taken: 231.5873 seconds
# num_workers = 4
# Time taken: 188.7113 seconds
# num_workers = 8
# Time taken: 185.3018 seconds
# --------------------------------------------------
# dataset_png_v2, dataset_version = 2
# num_workers = 2
# Time taken: 218.5305 seconds
# num_workers = 4
# Time taken: 181.3080 seconds
# num_workers = 8
# Time taken: 184.0530 seconds

