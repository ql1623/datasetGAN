import os
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import DataLoader
from skimage.util import view_as_windows
import h5py

from utils.config_reader import Config
import train_options as config

# config = Config('HiNet-master/utils/params.yaml')

def h5_to_tensor(data_path, sub_dir_name, shuffle):
    
    # h5_path = os.path.join(self.config.data['data_path'], 'modality_data_2d.h5')
    # h5 = h5py.File(h5_path, 'r')
    # np_dict = {} 
    
    # for key in h5_data.keys():
    #     np_dict[f"{key}_data"] = np.array(data[key])
        
    # data_dir_path = os.path.join(data_path, sub_dir_name)
    # for filename in os.listdir(data_dir_path):
    #     if filename.endswith('.h5'):
    #         if shuffle:
    #           h5_path = os.path.join(data_dir_path, filename)
    #         else:
    
    data_dir_path = os.path.join(data_path, sub_dir_name)
    if shuffle:
        for filename in os.listdir(data_dir_path):
            if filename.endswith('.h5') and 'shuffle' in filename:
                h5_path = os.path.join(data_dir_path, filename)
                print('data used was shuffled')
    else:
        for filename in os.listdir(data_dir_path):
            if filename.endswith('.h5') and 'shuffle' not in filename:
                h5_path = os.path.join(data_dir_path, filename)
                print('data used was not shuffled')
            
    h5_data = h5py.File(h5_path, 'r')
    data_dict = {} 
    for key in h5_data.keys():
        numpy_array = np.array(h5_data[key])
        data_dict[f"{key}_data"] = torch.tensor(numpy_array, dtype=torch.float32)
        
    return data_dict

# def get_patches(image, patch_size, num_patches):

#     h, w = image.shape
#     patch_size_h, patch_size_w = patch_size
#     num_patches_h, num_patches_w = num_patches
#     stride_h = (h-patch_size_h) // (num_patches_h - 1)
#     stride_w = (w-patch_size_w) // (num_patches_w - 1)
#     # overlap_h = patch_size-(h-patch_size)
#     # overlap_w = patch_size-(w-patch_size)
#     patches = np.array(torch.reshape((torch.tensor(image).unfold(0, patch_size_h, stride_h).unfold(1, patch_size_w, stride_w)), (num_patches_h*num_patches_w, patch_size_h, patch_size_w)))

    
#     return patches

def get_patches(image, patch_size, num_patches):

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
    
    
# def reconstruct_2d_image(patches, np_data_shape, stride):
#     h, w = np_data_shape
     
                  
class MRI_dataset(data.Dataset):
    def __init__(self, config, transform=None, train=True, test=False):
        self.config = config
        self.train = train
        self.test = test
        
        if train:
            shuffle = self.config.DATA_SHUFFLE
            self.data = h5_to_tensor(self.config.DATA_DIR, 'train', shuffle)
                    
        if test:
            shuffle = self.config.DATA_SHUFFLE
            self.data = h5_to_tensor(self.config.DATA_DIR, 'test', shuffle)
            
        self.num_samples =  self.data[list(self.data.keys())[0]].shape[0]
        # h5_path = os.path.join(self.config.data['data_path'], 'modality_data_2d.h5')
        # h5 = h5py.File(h5_path, 'r')
        # self.data = 
        # self.num_samples = self.data[list(self.data.keys())[0]].shape[0]
        
        
    def __getitem__(self, idx):
        # data_path = os.path.join(self.config.data['data_path'], 'modality_data_2d.h5')
        # t1, t1ce, t2, flair, seg = load_2d_data(data_path)
        patch_size = [128,128]
        num_patches = [2,2]
        
        t1_slice = get_patches(self.data['t1_data'][idx,:,:], patch_size, num_patches)
        t1ce_slice = get_patches(self.data['t1ce_data'][idx,:,:], patch_size, num_patches)
        t2_slice = get_patches(self.data['t2_data'][idx,:,:], patch_size, num_patches)
        flair_slice = get_patches(self.data['flair_data'][idx,:,:], patch_size, num_patches)
        seg_slice = get_patches(self.data['seg_data'][idx,:,:], patch_size, num_patches)
        
        # for modality, np_data in self.data.items():
        #     t1_data = np_data[idx,:,:]
        # t1_slice = t1[idx,:,:]
        # import pdb; pdb.set_trace()
        return t1_slice, t1ce_slice, t2_slice, flair_slice
    
    def __len__(self): 
        return self.num_samples
    

if __name__ == "__main__":
    # config = Config('HiNet-master/utils/params.yaml')
    print(config.BATCH_SIZE)
    train_data = MRI_dataset(config, train=True)
    print(len(train_data))
    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=False)
    t1_slice, t1ce_slice, t2_slice, flair_slice = next(iter(train_loader))
    print(t1_slice.shape)
    print(type(t1_slice))
    