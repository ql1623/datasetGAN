# import argparse
# import os
import torch

DATA_DIR = "/rds/general/user/ql1623/home/datasetGAN/data"
DATA_PNG_DIR = "/rds/general/user/ql1623/home/datasetGAN/data_png"

DATA_SHUFFLE = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_IDS = [0,1,2,3]

MODALITY_DIRECTION = "t1_t2_to_flair"
INPUT_MODALITIES = "t1_t2_flair"

BATCH_SIZE = 32
NUM_EPOCHS = 200

NUM_LAYERS = 4
NUM_FEATURES = 32
CONDITION_METHOD = "add"
GAN_VERSION = 2

LEARNING_RATE = 2e-4
LR_START_EPOCH = 50
LR_DECAY = 0.95

# NUM_WORKERS = 2
# IMAGE_SIZE = 256
LAMBDA_GAN_L1 = 1
LAMBDA_BCE = 0.1
LAMBDA_RECON = 1 
B1 = 0.5
B2 = 0.999
DROPOUT = 0.5

LOAD_MODEL = False
SAVE_MODEL = True

LOG_INTERVAL = 5
CHECKPOINT_INTERVAL = 5  

SAVE_RESULTS_DIR_NAME = "t1_t2_flair_cgan_v2_2"
# LOAD_RESULTS_DIR_NAME = "new_dataset"
SAVE_CHECKPOINT_DIR = "/rds/general/user/ql1623/home/datasetGAN/checkpoints"
SAVE_RESULTS_DIR = "/rds/general/user/ql1623/home/datasetGAN/results"




# class Options:
#     def __init__(self):
#         self.initialized = False

#     def initialize(self, parser):
#         parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
#         parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
        
#         parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
#         parser.add_argument('--lr_decay', type=float, default=0.95, help='Learning rate decay rate')
#         parser.add_argument('--lr_decay_epoch', type=int, default=100, help='Number of epoch to start learning rate decay')
        
#         parser.add_argument('--b1', type=float, default=0.5, help='Momentum of Adam optimizer')
#         parser.add_argument('--b2', type=float, default=0.999, help='Momentum of Adam optimizer')
        
#         parser.add_argument('--modalitydirection', type=str, default='t1t2toflair', help='Modality to translate into')
        
#         parser.add_argument('--checkpoint_interval', type=int, default=5, help='interval to save model')
#         parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save checkpoints and logs')
#         parser.add_argument('--results_dir', type=str, default='./results/', help='Directory to save test results.')
#         self.initialized = True
        
#         return parser

#     def gather_options(self):
#         if not self.initialized:
#             parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#             parser = self.initialize(parser)
            
#         self.parser = parser
#         return parser.parse_args()

#     def print_options(self, opt):
#         message = ''
#         message += '----------------- Options ---------------\n'
#         for k, v in sorted(vars(opt).items()):
#             comment = ''
#             default = self.parser.get_default(k)
#             if v != default:
#                 comment = '\t[default: %s]' % str(default)
#             message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
#         message += '----------------- End -------------------'
#         print(message)

#         # Save to the disk
#         os.makedirs(opt.save_dir, exist_ok=True)
#         file_name = os.path.join(opt.save_dir, 'opt.txt')
#         with open(file_name, 'wt') as opt_file:
#             opt_file.write(message)
#             opt_file.write('\n')

#     def parse(self):
#         opt = self.gather_options()
#         self.print_options(opt)
#         self.opt = opt
#         return self.opt
