import os
import numpy as np
import torch
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from PIL import Image
import random
import itertools
import time

from utils.utils import *

from skimage import exposure as ex

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

class Make_binary:
    def __call__(self, image):
        # make seg mask binary
        return torch.where(image != 0, torch.tensor(1.0), torch.tensor(0.0))
  
      
# def he(img):
#     img = np.array(img)
#     if(len(img.shape)==2):      #gray
#         outImg = ex.equalize_hist(img[:,:])*255 
#     elif(len(img.shape)==3):    #RGB
#         outImg = np.zeros((img.shape[0],img.shape[1],3))
#         for channel in range(img.shape[2]):
#             outImg[:, :, channel] = ex.equalize_hist(img[:, :, channel])*255

#     outImg[outImg>255] = 255
#     outImg[outImg<0] = 0
#     return outImg.astype(np.uint8)
def get_transform_params():
    # if we are resizing and crop
    new_h = new_w = 286
    crop_size = 256
    # i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(self.crop_size, self.crop_size))

    x = random.randint(0, np.maximum(0, new_w - crop_size))
    y = random.randint(0, np.maximum(0, new_h - crop_size))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}
    
def get_transforms(grayscale=True, params=None, resize_method=transforms.InterpolationMode.BICUBIC, convert=True, if_seg=False):
    import torchvision.transforms.functional as F
    transform_list = []
    resize_shape = [286, 286]
    crop_size = 256

    transform_list.append(transforms.Resize(resize_shape, resize_method))

    if params is None:
        transform_list.append(transforms.RandomCrop(crop_size))
        transform_list.append(transforms.RandomHorizontalFlip())
    else:
        transform_list.append(transforms.Lambda(lambda img: img.crop((params['crop_pos'][0], params['crop_pos'][1], params['crop_pos'][0] + crop_size, params['crop_pos'][1] + crop_size))))
        if params['flip']:
            transform_list.append(transforms.Lambda(lambda img: img.transpose(Image.FLIP_LEFT_RIGHT)))
    
    if convert:
        transform_list.append(transforms.ToTensor())
        if grayscale:   
            transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
               
    if if_seg:
        transform_list.append(Make_binary()) 

    return transforms.Compose(transform_list)



class MRI_dataset(Dataset):
    def __init__(self, input_modalities, data_png_dir, transform=True, train=True):
        # self.config = config
        self.input_modalities = input_modalities
        self.data_png_dir = data_png_dir
        self.transform = transform
        self.train = train
        
        modalities = self.input_modalities.split("_")
        mod_A, mod_B, mod_C = modalities[0], modalities[1], modalities[2]

        self.img_paths = {
            0: os.path.join(self.data_png_dir, mod_A, 'train' if train else 'test'),
            1: os.path.join(self.data_png_dir, mod_B, 'train' if train else 'test'),
            2: os.path.join(self.data_png_dir, mod_C, 'train' if train else 'test')
        }
        self.seg_img_paths = os.path.join(self.data_png_dir, "seg", 'train' if train else 'test')
        
        # self.img_lists = {
        #     'A': os.listdir(self.img_paths[0]),
        #     'B': os.listdir(self.img_paths[1]),
        #     'C': os.listdir(self.img_paths[2])
        # }
        self.img_lists = os.listdir(self.seg_img_paths)
        # assert os.listdir(self.img_paths[0]) == os.listdir(self.img_paths[1]), f"Images across {mod_A} and {mod_B} are not the same" 
        # assert os.listdir(self.img_paths[0]) == os.listdir(self.img_paths[2]), f"Images across {mod_A} and {mod_C} are not the same" 
        # assert os.listdir(self.img_paths[0]) == os.listdir(self.seg_img_paths), f"Images across {mod_A} and seg are not the same" 
        
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
        if not self.check_input_seq(self.input_modalities):
            print("Invalid input modalities format")
            return None, None, None, None
        
        patch_size = [128, 128]
        num_patches = [2, 2]
        
        img_id = self.img_lists[idx].split(".")[0]
        
        img_A_ori_idx, img_B_ori_idx, img_C_ori_idx = self.random_choice.choose()
        
        pil_A = Image.open(os.path.join(self.img_paths[img_A_ori_idx], self.img_lists[idx]))
        pil_B = Image.open(os.path.join(self.img_paths[img_B_ori_idx], self.img_lists[idx]))
        pil_C = Image.open(os.path.join(self.img_paths[img_C_ori_idx], self.img_lists[idx]))
        pil_seg = Image.open(os.path.join(self.seg_img_paths, self.img_lists[idx]))

        if self.transform:
            transform_params = get_transform_params()

            transform = get_transforms(grayscale=True, params=transform_params, if_seg=False)
            seg_transform = get_transforms(grayscale=False, params=transform_params, if_seg=True)
            
            tensor_A = transform(pil_A)
            tensor_B = transform(pil_B)
            tensor_C = transform(pil_C)
            tensor_seg = seg_transform(pil_seg)
        else:
            tensor_A = pil_A
            tensor_B = pil_B
            tensor_C = pil_C
            tensor_seg = pil_seg
        
        # in_out_comb = torch.tensor([img_A_ori_idx, img_B_ori_idx, img_C_ori_idx]).view(1,3).repeat(4, 1)
        in_out_comb = torch.tensor([img_A_ori_idx, img_B_ori_idx, img_C_ori_idx])

        return tensor_A, tensor_B, tensor_C, tensor_seg, in_out_comb, img_id
    
    def __len__(self): 
        return len(self.img_lists)

if __name__ == "__main__":
    def save_image_tensor(image_tensor, save_dir, file_name):
        """
        Saves an image tensor as a PNG file in the specified directory.

        Parameters:
        image_tensor (torch.Tensor): The image tensor to save.
        save_dir (str): The directory where the image will be saved.
        file_name (str): The name of the file to save the image as.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, file_name)
        save_image(image_tensor, save_path)

    print(config.BATCH_SIZE)
    print(config.INPUT_MODALITIES)

    start_time = time.time()

    train_data = MRI_dataset(config.INPUT_MODALITIES, config.DATA_PNG_DIR, transform=True, train=True)
    print(len(train_data))
    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=False, num_workers = 4)

    for index, images_labels in enumerate(train_loader):
        image_A, image_B, target_C, image_seg, in_out_comb, img_id = images_labels[0], images_labels[1], images_labels[2], images_labels[3], images_labels[4], images_labels[5]
        
        # save_dir = "/rds/general/user/ql1623/home/datasetGAN/pix_enhanced"
        # # Save images
        # for i in range(image_A.size(0)):
        #     save_image_tensor(image_A[i], save_dir, f"{img_id[i]}_A.png")
        #     save_image_tensor(image_B[i], save_dir, f"{img_id[i]}_B.png")
        #     save_image_tensor(target_C[i], save_dir, f"{img_id[i]}_C.png")
        #     save_image_tensor(image_seg[i], save_dir, f"{img_id[i]}_seg.png")
        break  
    print(in_out_comb.shape)
    target_labels = in_out_to_ohe_label(in_out_comb, 3)
    print(target_labels.shape)
    # print(f"Images saved to {save_dir}")
    print(f"Elapsed time: {time.time() - start_time} seconds")
 
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

