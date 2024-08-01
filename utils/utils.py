import torch
import torch.nn.init as init
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import os
import json
from collections import OrderedDict

import random
import itertools
from itertools import permutations
# import train_options as config

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse

from scipy.signal import gaussian

"""Train Utils used in training only"""

class GradientDifferenceLoss(nn.Module):
    def __init__(self, filter_type):
        super(GradientDifferenceLoss, self).__init__()
        if filter_type == "sobel":
            self.filter_h = torch.tensor([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) # make [1,1,3,3]
            self.filter_w = torch.tensor([[-1, -2, -1],[0, 0, 0],[1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        elif filter_type == "prewitt":  
            self.filter_h = torch.tensor([[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) # make [1,1,3,3]
            self.filter_w = torch.tensor([[-1, -1, -1],[0, 0, 0],[1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")
        
        # Register filters as buffers to ensure they are moved with the model
        # self.register_buffer('filter_h', filter_h)
        # self.register_buffer('filter_w', filter_w)

    def forward(self, pred, target):
        self.filter_h = self.filter_h.to(target.device)
        self.filter_w = self.filter_w.to(target.device)

        pred_grad_h = F.conv2d(pred, self.filter_h, padding=1)
        pred_grad_w = F.conv2d(pred, self.filter_w, padding=1)
        
        target_grad_h = F.conv2d(target, self.filter_h, padding=1)
        target_grad_w = F.conv2d(target, self.filter_w, padding=1)
        
        grad_diff_h = torch.mean((target_grad_h - pred_grad_h)**2)
        grad_diff_w = torch.mean((target_grad_w - pred_grad_w)**2)
        
        return grad_diff_h + grad_diff_w


# https://github.com/DCurro/CannyEdgePytorch/blob/master/net_canny.py
class CannyNet(nn.Module):
    def __init__(self, threshold=10.0, use_cuda=False):
        super(CannyNet, self).__init__()

        self.threshold = threshold
        self.use_cuda = use_cuda

        filter_size = 5
        generated_filters = gaussian(filter_size, std=1.0).reshape([1, filter_size])

        self.gaussian_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, filter_size), padding=(0, filter_size // 2))
        self.gaussian_filter_horizontal.weight.data.copy_(torch.from_numpy(generated_filters))
        self.gaussian_filter_horizontal.bias.data.copy_(torch.tensor([0.0], dtype=torch.float32))
        self.gaussian_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(filter_size, 1), padding=(filter_size // 2, 0))
        self.gaussian_filter_vertical.weight.data.copy_(torch.from_numpy(generated_filters.T))
        self.gaussian_filter_vertical.bias.data.copy_(torch.tensor([0.0], dtype=torch.float32))

        sobel_filter = torch.tensor([[1, 0, -1],
                                     [2, 0, -2],
                                     [1, 0, -1]], dtype=torch.float32)

        self.sobel_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0] // 2)
        self.sobel_filter_horizontal.weight.data.copy_(sobel_filter)
        self.sobel_filter_horizontal.bias.data.copy_(torch.tensor([0.0], dtype=torch.float32))
        self.sobel_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0] // 2)
        self.sobel_filter_vertical.weight.data.copy_(torch.transpose(sobel_filter, 0, 1))
        self.sobel_filter_vertical.bias.data.copy_(torch.tensor([0.0], dtype=torch.float32))

        filter_0 = np.array([[0, 0, 0], [0, 1, -1], [0, 0, 0]])
        filter_45 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]])
        filter_90 = np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]])
        filter_135 = np.array([[0, 0, 0], [0, 1, 0], [-1, 0, 0]])
        filter_180 = np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])
        filter_225 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 0]])
        filter_270 = np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]])
        filter_315 = np.array([[0, 0, -1], [0, 1, 0], [0, 0, 0]])

        all_filters = np.stack([filter_0, filter_45, filter_90, filter_135, filter_180, filter_225, filter_270, filter_315])

        self.directional_filter = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=filter_0.shape, padding=filter_0.shape[-1] // 2)
        self.directional_filter.weight.data.copy_(torch.from_numpy(all_filters[:, None, ...]))
        self.directional_filter.bias.data.copy_(torch.from_numpy(np.zeros(shape=(all_filters.shape[0],))))

        hysteresis = torch.tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=torch.float32) + 0.25
        self.hysteresis = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        with torch.no_grad():
            self.hysteresis.weight[:] = torch.from_numpy(hysteresis)

    def forward(self, img):
        device = img.device
        self.gaussian_filter_horizontal.to(device)
        self.gaussian_filter_vertical.to(device)
        self.sobel_filter_horizontal.to(device)
        self.sobel_filter_vertical.to(device)
        self.directional_filter.to(device)
        
        batch_size, _, height, width = img.size()

        # Noise reduction
        blur_horizontal = self.gaussian_filter_horizontal(img)
        blurred_img = self.gaussian_filter_vertical(blur_horizontal)

        # Gradient calculation
        grad_x_r = self.sobel_filter_horizontal(blurred_img)
        grad_y_r = self.sobel_filter_vertical(blurred_img)

        grad_mag = torch.sqrt(grad_x_r ** 2 + grad_y_r ** 2)
        grad_mag = (grad_mag / grad_mag.view(batch_size, -1).max(dim=1)[0].view(batch_size, 1, 1, 1)) * 255.0
        grad_orientation = (torch.atan2(grad_y_r, grad_x_r) * (180.0 / 3.14159)) + 180.0
        grad_orientation = torch.round(grad_orientation / 45.0) * 45.0

        # Non-maximum suppression
        all_filtered = self.directional_filter(grad_mag)

        indices_positive = (grad_orientation / 45) % 8
        indices_negative = ((grad_orientation / 45) + 4) % 8

        indices_positive = indices_positive.long().view(batch_size, -1)
        indices_negative = indices_negative.long().view(batch_size, -1)

        all_filtered_flat = all_filtered.view(batch_size, 8, -1)
        channel_select_filtered_positive = all_filtered_flat.gather(1, indices_positive.unsqueeze(1)).view(batch_size, 1, height, width)
        channel_select_filtered_negative = all_filtered_flat.gather(1, indices_negative.unsqueeze(1)).view(batch_size, 1, height, width)

        channel_select_filtered = torch.stack([channel_select_filtered_positive, channel_select_filtered_negative], dim=1)

        is_max = channel_select_filtered.min(dim=1)[0] > 0.0
        is_max = is_max.view(batch_size, 1, height, width)

        thin_edges = grad_mag.clone()
        thin_edges[is_max == 0] = 0.0

        # Double threshold
        
        # Edge Tracking by Hysteresis
        
        # thresholded = thin_edges.clone()
        # thresholded[thin_edges < self.threshold] = 0.0

        # early_threshold = grad_mag.clone()
        # early_threshold[grad_mag < self.threshold] = 0.0

        # if self.use_cuda:
        #     thresholded = thresholded.cuda()
        # weak = (thresholded == 0.5).float()
        # weak_is_high = (self.hysteresis(thresholded) > 1).float() * weak
        # thresholded = (thresholded >= 1).float() * 1 + weak_is_high * 1

        return thin_edges

class CannyEdge(nn.Module):
    def __init__(self, sigma=1, kernel_size=5, weak_pixel=75, strong_pixel=255, low_threshold=0.00392, high_threshold=0.15, use_hysteresis=True):
        super(CannyEdge, self).__init__()

        self.weak_pixel = weak_pixel
        self.strong_pixel = strong_pixel
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.use_hysteresis = use_hysteresis
        self.kernel_size = kernel_size
        self.sigma = sigma
        
    def noise_removal(self, kernel_size, sigma, img):
        device = img.device
        generated_filters = gaussian(kernel_size, std=sigma).reshape([1, kernel_size]).astype(np.float32)
        gaussian_filter_horizontal = torch.from_numpy(generated_filters).unsqueeze(0).unsqueeze(0).to(device)
        gaussian_filter_vertical = torch.from_numpy(generated_filters.T).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            smoothed_img = F.conv2d(img, gaussian_filter_horizontal, padding=(0, kernel_size // 2))
            smoothed_img = F.conv2d(smoothed_img, gaussian_filter_vertical, padding=(kernel_size // 2, 0))
            
        return smoothed_img

    def grad_calc(self, smoothed_img):
        B = smoothed_img.shape[0]
        device = smoothed_img.device
        
        sobel_filter = torch.tensor([[1, 0, -1],
                                     [2, 0, -2],
                                     [1, 0, -1]], dtype=torch.float32).to(device)
        sobel_filter_horizontal = sobel_filter.unsqueeze(0).unsqueeze(0).to(device)
        sobel_filter_vertical = sobel_filter.t().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            Ix = F.conv2d(smoothed_img, sobel_filter_horizontal, padding=1)
            Iy = F.conv2d(smoothed_img, sobel_filter_vertical, padding=1)

        grad_mag = torch.sqrt(Ix ** 2 + Iy ** 2)
        grad_mag /= grad_mag.view(B, -1).max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        # grad_mag = grad_mag / (grad_mag.view(B, 1, -1).max(dim=2)[0].view(B, 1, 1, 1))
        grad_direction = (torch.atan2(Iy, Ix) * (180.0 / 3.1415926)) + 180
        grad_direction = torch.round(grad_direction / 45.0) * 45.0
        
        return (grad_mag, grad_direction)
    
    def non_max_suppression(self, grad_mag, grad_direction):
        B, C, H, W = grad_mag.shape
        device = grad_mag.device
        # filters were flipped manually
        filter_0 = np.array([[0, 0, 0],
                            [0, 1, -1],
                            [0, 0, 0]], dtype=np.float32)

        filter_45 = np.array([[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, -1]], dtype=np.float32)

        filter_90 = np.array([[0, 0, 0],
                            [0, 1, 0],
                            [0, -1, 0]], dtype=np.float32)

        filter_135 = np.array([[0, 0, 0],
                            [0, 1, 0],
                            [-1, 0, 0]], dtype=np.float32)

        filter_180 = np.array([[0, 0, 0],
                            [-1, 1, 0],
                            [0, 0, 0]], dtype=np.float32)

        filter_225 = np.array([[-1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]], dtype=np.float32)

        filter_270 = np.array([[0, -1, 0],
                            [0, 1, 0],
                            [0, 0, 0]], dtype=np.float32)

        filter_315 = np.array([[0, 0, -1],
                            [0, 1, 0],
                            [0, 0, 0]], dtype=np.float32)

        all_filters = np.stack([filter_0, filter_45, filter_90, filter_135, filter_180, filter_225, filter_270, filter_315])
        directional_filter = torch.from_numpy(all_filters).unsqueeze(1).to(device)
        with torch.no_grad():
            all_filtered = F.conv2d(grad_mag, directional_filter, padding=1)

        indices_positive = (grad_direction / 45).long() % 8
        indices_negative = (indices_positive + 4) % 8

        indices_positive = indices_positive.view(B, 1, -1)
        indices_negative = indices_negative.view(B, 1, -1)

        all_filtered_flat = all_filtered.view(B, 8, -1)
        channel_select_filtered_positive = all_filtered_flat.gather(1, indices_positive.view(B, 1, -1)).view(B, 1, H, W)
        channel_select_filtered_negative = all_filtered_flat.gather(1, indices_negative.view(B, 1, -1)).view(B, 1, H, W)

        # is_max = torch.min(channel_select_filtered_positive, channel_select_filtered_negative) == grad_mag

        # thin_edges = grad_mag.clone()
        # thin_edges[~is_max] = 0.0
        channel_select_filtered = torch.stack([channel_select_filtered_positive, channel_select_filtered_negative], dim=1)

        is_max = channel_select_filtered.min(dim=1)[0] > 0.0
        is_max = is_max.view(B, 1, H, W)

        thin_edges = grad_mag.clone()
        thin_edges[is_max == 0] = 0.0  # [1,1,h,w]
        
        return thin_edges

    def hysteresis(self, intermediate_threshold):
        device = intermediate_threshold.device
        
        strong_neighbours = torch.tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=torch.float32).to(device)
        strong_filter = strong_neighbours.unsqueeze(0).unsqueeze(0).to(device)
        while True:
                weak_mask = intermediate_threshold == self.weak_pixel
                strong_neighbors = F.conv2d((intermediate_threshold == self.strong_pixel).float(), strong_filter, padding=1)
                new_strong = weak_mask & (strong_neighbors > 0)
                if not new_strong.any():
                    break
                intermediate_threshold[new_strong] = self.strong_pixel

        final_edges = intermediate_threshold.clone()
        final_edges[final_edges != self.strong_pixel] = 0
        
        return final_edges
    
    def forward(self, img):
        B, C, H, W = img.shape

        device = img.device

        # Noise Reduction
        smoothed_img = self.noise_removal(self.kernel_size, self.sigma, img)

        # Gradient calculation
        grad_mag, grad_direction = self.grad_calc(smoothed_img)
        
        # Non-maximum suppression
        thin_edges = self.non_max_suppression(grad_mag, grad_direction)
        
        # Double threshold
        high_threshold = thin_edges.max() * self.high_threshold

        strong_edges = thin_edges >= high_threshold
        weak_edges = (thin_edges < high_threshold) & (thin_edges >= self.low_threshold)

        intermediate_threshold = torch.zeros_like(thin_edges)
        intermediate_threshold[strong_edges] = self.strong_pixel
        intermediate_threshold[weak_edges] = self.weak_pixel

        # Hysteresis
        if self.use_hysteresis:
            final_edges = self.hysteresis(intermediate_threshold)
            
        else:
            final_edges = intermediate_threshold[strong_edges]

        # return smoothed_img, grad_mag, grad_direction, thin_edges, intermediate_threshold, final_edges
        return final_edges


class GradientDifferenceLossCanny(nn.Module):
    def __init__(self, low_threshold=0.00392, high_threshold=0.15):# , use_cuda=False
        super(GradientDifferenceLossCanny, self).__init__()
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        # self.use_cuda = use_cuda
        self.canny_net = CannyEdge(sigma=1, kernel_size=5, weak_pixel=75, strong_pixel=255, low_threshold=self.low_threshold, high_threshold=self.high_threshold, use_hysteresis=True)
        
    def forward(self, pred, target):
        # Initialize lists to hold edge maps for each image in the batch
        pred_edges = self.canny_net(pred)
        target_edges = self.canny_net(target)
        
        grad_diff = torch.mean((target_edges - pred_edges) ** 2)
        return grad_diff

class GradientDifferenceLossCanny_v1(nn.Module):
    def __init__(self, low_threshold=50, high_threshold=150):
        super(GradientDifferenceLossCanny_v1, self).__init__()
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def canny_edge_detection(self, img):
        # Convert PyTorch tensor to numpy array
        img_np = img.squeeze().cpu().numpy()

        # Apply Canny edge detection
        edges = cv2.Canny(img_np, self.low_threshold, self.high_threshold)

        # Convert edges back to PyTorch tensor
        edges_tensor = torch.tensor(edges, dtype=torch.float32).unsqueeze(0).to(img.device)
        return edges_tensor

    def forward(self, pred, target):
        # Initialize lists to hold edge maps for each image in the batch
        pred_edges_list = []
        target_edges_list = []

        # Loop through each image in the batch
        for i in range(pred.size(0)):  # Batch size
            for j in range(pred.size(1)):  # Number of channels
                pred_edges = self.canny_edge_detection(pred[i, j, :, :].unsqueeze(0))
                target_edges = self.canny_edge_detection(target[i, j, :, :].unsqueeze(0))
                pred_edges_list.append(pred_edges)
                target_edges_list.append(target_edges)

        # Stack the lists to form tensors of shape [B, C, H, W]
        pred_edges = torch.stack(pred_edges_list).view(pred.size())
        target_edges = torch.stack(target_edges_list).view(target.size())

        # Calculate the gradient difference loss
        grad_diff = torch.mean((pred_edges - target_edges) ** 2)
        return grad_diff


class L1LossWithAttention(nn.Module):
    def __init__(self):
        super(L1LossWithAttention, self).__init__()
        self.loss_fn = nn.L1Loss(reduction='none')

    def forward(self, pred, target, seg_output):
        # attention_map = torch.sigmoid(seg_output)
        # l1_loss = self.loss_fn(pred, target)
        weighted_l1_loss = seg_output * self.loss_fn(pred, target)
        return weighted_l1_loss.mean()


class L2LossWithAttention(nn.Module):
    def __init__(self):
        super(L2LossWithAttention, self).__init__()
        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, pred, target, seg_output):
        # attention_map = torch.sigmoid(seg_output)
        # l1_loss = self.loss_fn(pred, target)
        weighted_l2_loss = seg_output * self.loss_fn(pred, target)
        return weighted_l2_loss.mean()

class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super (DiceLoss, self).__init__()
        self.epsilon = epsilon
        
    def forward(self, image_fake, image_real):
        # image_fake and image_real should be squeezed = [B, C=1, H, W] --> [B, H, W]
        intersection = (image_fake * image_real).sum(dim=(-1,-2))
        union = (image_fake).sum(dim=(-1,-2)) + (image_real).sum(dim=(-1,-2))
        union = torch.where(union == 0, intersection, union)
        dice_coeff = (2 * intersection + self.epsilon) / (union + self.epsilon)
        return 1 - dice_coeff.mean()

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
        print("Training Options saved to: ", save_path)
                    
    else:
        save_path = os.path.join(file_dir, dir_name + "_test")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        with open(os.path.join(save_path, "test_opt.txt"), 'a') as file:
            file.write(f"================ Testing Options {run_id} ================\n")
            for key, value in config.__dict__.items():
                if not key.startswith('__') and not callable(value) and not isinstance(value, type(config)):
                    file.write(f"{key}: {value}\n")
        print("Testing Options saved to: ", save_path)
        
        
def save_options(options, run_id, file_dir, dir_name, train=True):
    if train:
        save_path = os.path.join(file_dir, "chkpt_" + dir_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        with open(os.path.join(save_path, "train_opt.txt"), 'a') as file:
            file.write(f"================ Training Options {run_id} ================\n")
            for arg, value in vars(options).items():
                file.write(f"{arg}: {value}\n")
        print("Training Options saved to: ", save_path)
                    
    else:
        save_path = os.path.join(file_dir, dir_name + "_test")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        with open(os.path.join(save_path, "test_opt.txt"), 'a') as file:
            file.write(f"================ Testing Options {run_id} ================\n")
            for arg, value in vars(options).items():
                file.write(f"{arg}: {value}\n")
        print("Testing Options saved to: ", save_path)


             
def log_loss_to_txt(log_dir, dir_name, run_id, epoch, loss_data, loss_name):
    # losses now: loss_G_BCE, loss_G_L1, loss_G_reconA, loss_G_reconB, loss_G_GDL, loss_D_fake, loss_D_real, loss_S
    save_path = os.path.join(log_dir, "chkpt_" + dir_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # num_loss = len(loss_data)
    # loss_G_BCE, loss_G_L1, loss_G_reconA, loss_G_reconB, loss_G_GDL, loss_D_fake, loss_D_real, loss_S = loss_data   
    # "G_BCE", "G_L1", "G_reA", "G_reB", "G_GDL", "D_fake", "D_real", "S_fake" = loss_name
    log_file_path = os.path.join(save_path, "loss_log.txt")
    
    with open(log_file_path, 'a') as file:
        if epoch == 1:
            # Write header if the it is start of training
            file.write(f"================ Training Loss {run_id} ================\n")
            header = "epoch:\t" + loss_name[0] + "\t\t" + "\t\t".join(loss_name[1:]) + "\n"
            file.write(header)
            # file.write(f"epoch:\tG_BCE\t\tG_L1\t\tG_reA\t\tG_reB\t\tG_GDL\t\tD_fake\t\tD_real\n")
        # file.write(f"{epoch}\t\t{loss_G_BCE:.6f}\t{loss_G_L1:.6f}\t{loss_G_reconA:.6f}\t{loss_G_reconB:.6f}\t{loss_G_GDL:.6f}\t{loss_D_fake:.6f}\t{loss_D_real:.6f}\n")
        loss_values = [f"{loss:.6f}" for loss in loss_data]
        line = f"{epoch}" + "\t\t" + "\t".join(loss_values) + "\n"
        file.write(line)
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

def load_checkpoint(model, optimizer, lr, checkpoint_dir, dir_name, epoch_num, model_type="gen"):
    save_dir = os.path.join(checkpoint_dir, "chkpt_" + dir_name)
    if model_type=="gen":
        save_path = os.path.join(save_dir, f"{epoch_num}_net_G.pth")
        print("Loaded gen from: ", save_path)
    elif model_type=="disc":
        save_path = os.path.join(save_dir, f"{epoch_num}_net_D.pth")
        print("Loaded disc from: ", save_path)
    elif model_type=="seg":
        save_path = os.path.join(save_dir, f"{epoch_num}_net_S.pth")
        print("Loaded seg from: ", save_path)
    else:
        raise Exception("model_type defined in wrong format")
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

"""Test Utils used in testing only"""

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


def evaluate_images(pred_images, real_images, run_id, batch_index, eval_log_dir, dir_name): # pred_images and real_images [32,1,200,200]
    
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
            eval_log_path = os.path.join(eval_log_dir, dir_name + "_test")
        else:
            eval_log_path = os.path.join(eval_log_dir, "unnamed_test")
            print("results folder name not defined, saved test results in: ", eval_log_path)
        if not os.path.exists(eval_log_path):
            os.makedirs(eval_log_path)

        # saving json file
        metrics_json_path = os.path.join(eval_log_path, "test_results.json")
        if os.path.exists(metrics_json_path):
            with open(metrics_json_path, "r") as file:
                test_logs = json.load(file)
        else:
            # test_logs = OrderedDict()
            test_logs = {}
        if run_id not in test_logs:
            test_logs[run_id] = []
        
        test_logs[run_id].append({"SSIM": avg_ssim, "PSNR": avg_psnr, "MSE": avg_mse})
        
        with open(metrics_json_path, 'w') as file:
            json.dump(test_logs, file, indent=4)
        
        # saving txt file
        eval_log_file = os.path.join(eval_log_path, "test_results.txt")
        
        with open(eval_log_file, 'a') as log_file:
            if batch_index == 0:
                log_file.write(f"================ Testing Model {run_id} ================\n")
            log_file.write("[" + f"Batch {batch_index+1}: MSE: {avg_mse:.6f} | SSIM: {avg_ssim:.6f} | PSNR: {avg_psnr:.6f}" + "]\n")

    return avg_ssim, avg_psnr, avg_mse, error_metrics


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


def save_results_ori(img_id, image_A, image_B, pred_images, real_images, save_results_dir, dir_name):
    """Save images to a specified folder."""
    save_path = os.path.join(save_results_dir, dir_name + "_test", "images")
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    # need to add to output what output modality is, maybe use config.INPUT_MODALITIES to save the other 2 
    for index, img_filename in enumerate(img_id):
        id_str = img_filename
        
        save_image(image_A[index], os.path.join(save_path, f"{id_str}_real_A.png"))
        save_image(image_B[index], os.path.join(save_path, f"{id_str}_real_B.png"))
        save_image(pred_images[index], os.path.join(save_path, f"{id_str}_fake_C.png"))
        save_image(real_images[index], os.path.join(save_path, f"{id_str}_real_C.png"))
        
            
def save_results(run_id, img_id, in_out_comb, input_mod, batch_index, image_A, image_B, pred_images, real_images, error_metrics, save_results_dir, dir_name):
    """Save images to a specified folder."""
    save_path = os.path.join(save_results_dir, dir_name + "_test", "images")
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    detailed_metrics_json_path = os.path.join(save_results_dir, dir_name + "_test", "test_results_detailed.json")
    if os.path.exists(detailed_metrics_json_path):
        with open(detailed_metrics_json_path, "r") as file:
            detailed_test_logs = json.load(file)
    else:
        detailed_test_logs = {}
    if run_id not in detailed_test_logs:
        detailed_test_logs[run_id] = []
              
    batch_detailed_results = []
    
    modalities = input_mod.split("_")
    modalities_map = {
        0: modalities[0],
        1: modalities[1],
        2: modalities[2],
    }
    batch_in_out = in_out_comb[::4].cpu().numpy()
    # import pdb; pdb.set_trace()
    # need to add to output what output modality is, maybe use config.INPUT_MODALITIES to save the other 2 
    for index, img_filename in enumerate(img_id):
        id_str = img_filename
        mod_A, mod_B, mod_C = modalities_map[batch_in_out[index][0]], modalities_map[batch_in_out[index][1]], modalities_map[batch_in_out[index][2]]
        
        save_image(image_A[index], os.path.join(save_path, f"{id_str}_real_A_{mod_A}.png"))
        save_image(image_B[index], os.path.join(save_path, f"{id_str}_real_B_{mod_B}.png"))
        save_image(pred_images[index], os.path.join(save_path, f"{id_str}_fake_C_{mod_C}.png"))
        save_image(real_images[index], os.path.join(save_path, f"{id_str}_real_C_{mod_C}.png"))
        
        img_filename_and_comb = "_".join([id_str, mod_A, mod_B, "to", mod_C])
        img_error_metrics = {
            "SSIM": error_metrics["SSIM"][index],
            "PSNR": error_metrics["PSNR"][index],
            "MSE": error_metrics["MSE"][index],
        }
        
        batch_detailed_results.append({"Img_details": img_filename_and_comb, "Error_metrics": img_error_metrics})
        
    batch_results = {
        f"Batch_{batch_index}": batch_detailed_results
    }    
    detailed_test_logs[run_id].append(batch_results)

    with open(detailed_metrics_json_path, 'w') as file:
        json.dump(detailed_test_logs, file, indent=4)

def save_results_seg(run_id, img_id, in_out_comb, input_mod, batch_index, 
                     image_A, image_B, pred_images, real_images, pred_seg, real_seg, error_metrics, save_results_dir, dir_name):
    """Save images to a specified folder."""
    save_path = os.path.join(save_results_dir, dir_name + "_test", "images")
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    detailed_metrics_json_path = os.path.join(save_results_dir, dir_name + "_test", "test_results_detailed.json")
    if os.path.exists(detailed_metrics_json_path):
        with open(detailed_metrics_json_path, "r") as file:
            detailed_test_logs = json.load(file)
    else:
        detailed_test_logs = {}
    if run_id not in detailed_test_logs:
        detailed_test_logs[run_id] = []
              
    batch_detailed_results = []
    
    modalities = input_mod.split("_")
    modalities_map = {
        0: modalities[0],
        1: modalities[1],
        2: modalities[2],
    }
    batch_in_out = in_out_comb.cpu().numpy()
    # import pdb; pdb.set_trace()
    # need to add to output what output modality is, maybe use config.INPUT_MODALITIES to save the other 2 
    for index, img_filename in enumerate(img_id):
        id_str = img_filename
        mod_A, mod_B, mod_C = modalities_map[batch_in_out[index][0]], modalities_map[batch_in_out[index][1]], modalities_map[batch_in_out[index][2]]
        
        save_image(image_A[index], os.path.join(save_path, f"{id_str}_real_A_{mod_A}.png"))
        save_image(image_B[index], os.path.join(save_path, f"{id_str}_real_B_{mod_B}.png"))
        save_image(pred_images[index], os.path.join(save_path, f"{id_str}_fake_C_{mod_C}.png"))
        save_image(real_images[index], os.path.join(save_path, f"{id_str}_real_C_{mod_C}.png"))
        save_image(pred_seg[index], os.path.join(save_path, f"{id_str}_fake_seg_C_{mod_C}.png"))
        save_image(real_seg[index], os.path.join(save_path, f"{id_str}_real_seg_C_{mod_C}.png"))
        
        img_filename_and_comb = "_".join([id_str, mod_A, mod_B, "to", mod_C])
        img_error_metrics = {
            "SSIM": error_metrics["SSIM"][index],
            "PSNR": error_metrics["PSNR"][index],
            "MSE": error_metrics["MSE"][index],
        }
        
        batch_detailed_results.append({"Img_details": img_filename_and_comb, "Error_metrics": img_error_metrics})
        
    batch_results = {
        f"Batch_{batch_index}": batch_detailed_results
    }    
    detailed_test_logs[run_id].append(batch_results)

    with open(detailed_metrics_json_path, 'w') as file:
        json.dump(detailed_test_logs, file, indent=4)


def generate_html_seg(run_id, save_results_dir, dir_name):
    """Generate an HTML file to view the images."""
    save_images_path = os.path.join(save_results_dir, dir_name + "_test", "images")
    save_html_path = os.path.join(save_results_dir, dir_name + "_test")
    html_content = f"<html><body><h2>Testing Results {dir_name}</h2>"
    pred_images = {}
    
    for filename in sorted(os.listdir(save_images_path)):
        if filename.endswith(".png"):
            img_id = "_".join(filename.split("_")[:-3])
            # print(img_id)
            if img_id not in pred_images:
                pred_images[img_id] = []
            pred_images[img_id].append(filename)
    
    # print(pred_images.keys())        
    for img_id, files in pred_images.items():
        html_content += f'<h3>{img_id}</h3>'
        html_content += '<div style="display: flex;">'
        img_sequence = ['real_A', 'real_B', 'fake_C', 'real_C', 'fake_seg_C', 'real_seg_C']
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
        
def generate_html(run_id, save_results_dir, dir_name):
    """Generate an HTML file to view the images."""
    save_images_path = os.path.join(save_results_dir, dir_name + "_test", "images")
    save_html_path = os.path.join(save_results_dir, dir_name + "_test")
    html_content = f"<html><body><h2>Testing Results {dir_name}</h2>"
    pred_images = {}
    
    for filename in sorted(os.listdir(save_images_path)):
        if filename.endswith(".png"):
            img_id = "_".join(filename.split("_")[:-3])
            # print(img_id)
            if img_id not in pred_images:
                pred_images[img_id] = []
            pred_images[img_id].append(filename)
    
    # print(pred_images.keys())        
    for img_id, files in pred_images.items():
        html_content += f'<h3>{img_id}</h3>'
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

def generate_html_ori(run_id, save_results_dir, dir_name):
    """Generate an HTML file to view the images."""
    save_images_path = os.path.join(save_results_dir, dir_name + "_test", "images")
    save_html_path = os.path.join(save_results_dir, dir_name + "_test")
    html_content = f"<html><body><h2>Testing Results {dir_name}</h2>"
    pred_images = {}
    
    for filename in sorted(os.listdir(save_images_path)):
        if filename.endswith(".png"):
            img_id = "_".join(filename.split("_")[:-2])
            # print(img_id)
            if img_id not in pred_images:
                pred_images[img_id] = []
            pred_images[img_id].append(filename)
    
    # print(pred_images.keys())        
    for img_id, files in pred_images.items():
        html_content += f'<h3>{img_id}</h3>'
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



"""Base Utils used in both train and test"""

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


def get_key(value, mod_dict):
    for k, v in mod_dict.items():
        if value == v:
            return k
    return "Corresponding key not found for this value"


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
            self.remains - 1
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
    
def get_data_for_input_mod(images, input_modalities):
    # images from dataloader are in: [image_A, image_B, image_C], images with size [batch_size, 4, 128, 128]
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
        pass
    else:
        # raise ValueError("Invalid input modalities format")
        print("Invalid input modalities format")
        
    modalities = input_modalities.split("_") # something like "t1_t1ce_t2"
    # modalities_map = {
    #     modalities[0]: 0,
    #     modalities[1]: 1,
    #     modalities[2]: 2
    # }
        
    in_out_combinations = [
        (0, 1, 2),
        (0, 2, 1),
        (1, 2, 0)
    ]
    
    batch_size = images[0].shape[0]
    random_choice = BalancedChooser(in_out_combinations)
    image_A_list, image_B_list, image_C_list, target_labels_list, target_name_list = [], [], [], [], []
    for i in range(batch_size):
        img_A_mod_num, img_B_mod_num, img_C_mod_num = random_choice.choose()
                    
        image_A_list.append(images[img_A_mod_num][i])
        image_B_list.append(images[img_B_mod_num][i])
        image_C_list.append(images[img_C_mod_num][i])
        target_labels_list.append(torch.tensor([img_C_mod_num] * images[img_C_mod_num].shape[1]))
        # target_name_list.append(get_key(img_C_mod_num, modalities_map))
        
    images_A_batch = torch.unsqueeze(torch.cat(image_A_list, dim=0), dim=1).type(torch.FloatTensor)
    images_B_batch = torch.unsqueeze(torch.cat(image_B_list, dim=0), dim=1).type(torch.FloatTensor)
    images_C_batch = torch.unsqueeze(torch.cat(image_C_list, dim=0), dim=1).type(torch.FloatTensor)
    target_labels_batch = get_ohe_label_vec(torch.cat(target_labels_list, dim=0), 3)
    # print(target_labels_batch.shape)
    
    return images_A_batch, images_B_batch, images_C_batch, target_labels_batch #, target_name_list

def get_ohe_label_vec(output_modality_num, num_classes):
    return F.one_hot(output_modality_num, num_classes)

def in_out_to_ohe_label(in_out_combination, num_classes):
    # import pdb; pdb.set_trace()
    output_modality_num = torch.tensor([comb[num_classes-1] for comb in in_out_combination]) # get last element in in_out_combination
    return F.one_hot(output_modality_num, num_classes) # ohe

def get_data_for_input_mod_h5(images, input_modalities):
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
        counts = {comb: 0 for comb in in_out_combinations}
        image_A_list, image_B_list, image_C_list, target_labels_list, target_name_list = [], [], [], [], []
        for i in range(batch_size):
            min_count = min(counts.values())
            possible_choices = [comb for comb in in_out_combinations if counts[comb] == min_count]
            img_A_mod_num, img_B_mod_num, img_C_mod_num = random.choice(possible_choices)
            counts[(img_A_mod_num, img_B_mod_num, img_C_mod_num)] += 1
            
            image_A_list.append(images[img_A_mod_num][i])
            image_B_list.append(images[img_B_mod_num][i])
            image_C_list.append(images[img_C_mod_num][i])
            target_labels_list.append(torch.tensor([img_C_mod_num] * images[img_C_mod_num].shape[1]))
            target_name_list.append(get_key(img_C_mod_num, modalities_map))
            
        images_A_batch = torch.unsqueeze(torch.cat(image_A_list, dim=0), dim=1).type(torch.FloatTensor)
        images_B_batch = torch.unsqueeze(torch.cat(image_B_list, dim=0), dim=1).type(torch.FloatTensor)
        images_C_batch = torch.unsqueeze(torch.cat(image_C_list, dim=0), dim=1).type(torch.FloatTensor)
        target_labels_batch = get_ohe_label_vec(torch.cat(target_labels_list, dim=0), len(modalities_map), images_C_batch.shape[2])
        # print(target_labels_batch.shape)
    else:
        # raise ValueError("Invalid input modalities format")
        print("Invalid input modalities format")
    
    return images_A_batch, images_B_batch, images_C_batch, target_labels_batch, target_name_list

def reshape_data(image_A, image_B, image_C, target_labels):
    # [batch_size, patches, 128, 128] --> [batch_size*patches, 1, 128, 128]
    image_A = image_A.view(-1, 1, 128, 128)
    image_B = image_B.view(-1, 1, 128, 128)
    image_C = image_C.view(-1, 1, 128, 128)
    target_labels = target_labels.view(-1, 3)
    return image_A, image_B, image_C, target_labels

def reshape_data_seg(image_A, image_B, image_C, image_seg, target_labels):
    # [batch_size, patches, 128, 128] --> [batch_size*patches, 1, 128, 128]
    image_A = image_A.view(-1, 1, 128, 128)
    image_B = image_B.view(-1, 1, 128, 128)
    image_C = image_C.view(-1, 1, 128, 128)
    image_seg = image_seg.view(-1, 1, 128, 128)
    target_labels = target_labels.view(-1, 3)
    return image_A, image_B, image_C, image_seg, target_labels
    
# def get_condition_label(modality_direction, num_classes):
#     output_mod = modality_direction.split("_")[-1]
#     out_mod_ohe_label = get_ohe_label(output_mod, num_classes)
    
#     return out_mod_ohe_label
    
if __name__ == "__main__":
    from datasetGAN import train_options as config
    from torch.utils.data import DataLoader
    from datasetGAN.dataset import MRI_dataset
    
    print(config.BATCH_SIZE)
    train_data = MRI_dataset(config, train=True)
    print(len(train_data))
    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=False)
    
    for index, images in enumerate(train_loader):
        image_A, image_B, target_C, target_labels, target_name_list = get_data_for_input_mod(images, config.INPUT_MODALITIES)
        print(target_labels.shape)
        break

    