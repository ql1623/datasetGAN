import os
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import DataLoader
from skimage.util import view_as_windows
import h5py
import torchvision.transforms as transforms

import time
from utils.utils import *

from utils.config_reader import Config
import train_options as config

# config = Config('HiNet-master/utils/params.yaml')

def get_patches(image, patch_size, num_patches):
    image = image.squeeze(0)
    h, w = image.shape
    patch_size_h, patch_size_w = patch_size
    num_patches_h, num_patches_w = num_patches
    stride_h = (h-patch_size_h) // (num_patches_h - 1)
    stride_w = (w-patch_size_w) // (num_patches_w - 1)
    # overlap_h = patch_size-(h-patch_size)
    # overlap_w = patch_size-(w-patch_size)
    # patches = np.reshape(np.lib.stride_tricks.sliding_window_view(image, patch_size)[::stride_h, ::stride_w], (num_patches_h*num_patches_w, patch_size_h, patch_size_w))
    patches = []
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            patch_index_h = i * stride_h
            patch_index_w = j * stride_w
            patch = image[patch_index_h:patch_index_h + patch_size_h, patch_index_w:patch_index_w + patch_size_w]
            patches.append(patch)
    
    return torch.stack(patches)
    
    
                  
class MRI_dataset(data.Dataset):
    def __init__(self, config, transform=True, train=True):
        self.config = config
        self.train = train
        
        modalities = config.INPUT_MODALITIES.split("_")
        mod_A, mod_B, mod_C = modalities[0], modalities[1], modalities[2]
        
        self.img_paths = {
            'A': os.path.join(config.DATA_PNG_DIR, mod_A, 'train' if train else 'test'),
            'B': os.path.join(config.DATA_PNG_DIR, mod_B, 'train' if train else 'test'),
            'C': os.path.join(config.DATA_PNG_DIR, mod_C, 'train' if train else 'test')
        }
        self.img_lists = {
            'A': os.listdir(self.img_paths['A']),
            'B': os.listdir(self.img_paths['B']),
            'C': os.listdir(self.img_paths['C'])
        }
        
        if transform:
            self.transform = transforms.Compose([
                transforms.ToTensor(), # already normalize the image to [0, 1]
                transforms.Normalize(mean=[0.5], std=[0.5]) # then normalize the image to [-1, 1]
            ])        
        
    def __getitem__(self, idx):
        # data_path = os.path.join(self.config.data['data_path'], 'modality_data_2d.h5')
        # t1, t1ce, t2, flair, seg = load_2d_data(data_path)
        patch_size = [128,128]
        num_patches = [2,2]
        
        pat_id = self.img_lists["A"][idx].split(".")
        pil_A = Image.open(os.path.join(self.img_paths["A"], self.img_lists["A"][idx]))
        pil_B = Image.open(os.path.join(self.img_paths["B"], self.img_lists["B"][idx]))
        pil_C = Image.open(os.path.join(self.img_paths["C"], self.img_lists["C"][idx]))
        tensor_A = self.transform(pil_A)
        tensor_B = self.transform(pil_B)
        tensor_C = self.transform(pil_C)
        
        
        image_A = get_patches(tensor_A, patch_size, num_patches)
        image_B = get_patches(tensor_B, patch_size, num_patches)
        image_C = get_patches(tensor_C, patch_size, num_patches)
        
        # for modality, np_data in self.data.items():
        #     t1_data = np_data[idx,:,:]
        # t1_slice = t1[idx,:,:]
        # import pdb; pdb.set_trace()
        return image_A, image_B, image_C, pat_id
    
    def __len__(self): 
        return len(self.img_lists["A"])
    

if __name__ == "__main__":
    start_time = time.time()
    print(config.BATCH_SIZE)
    print(config.INPUT_MODALITIES)
    train_data = MRI_dataset(config, train=True)
    print(len(train_data))
    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=False)
    # image_A, image_B, image_C, pat_id = next(iter(train_loader))
    # print(image_A.shape)
    # print(type(image_A))
    # print(type(pat_id))
    # print(pat_id.shape)

    for index, images in enumerate(train_loader):
        # image_A, image_B, target_C, target_labels, target_name_list = get_data_for_input_mod(images, config.INPUT_MODALITIES)
        image_A, image_B, target_C, target_labels = get_data_for_input_mod(images, config.INPUT_MODALITIES)
        print(target_labels.shape)
        # print(target_name_list)
        break
    
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.4f} seconds")