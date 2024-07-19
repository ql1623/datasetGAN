import torch
import argparse
import os

class TrainOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Training Options")
        self._initialize()

    def _initialize(self):
        self.parser.add_argument('--data_dir', type=str, default="/rds/general/user/ql1623/home/datasetGAN/data")
        self.parser.add_argument('--data_png_dir', type=str, default="/rds/general/user/ql1623/home/datasetGAN/data_png")
        self.parser.add_argument('--data_shuffle', type=bool, default=False)
        self.parser.add_argument('--device', type=str, default="cpu")
        self.parser.add_argument('--gpu_ids', type=str, default="0")
        
        self.parser.add_argument('--modality_direction', '--mod_direction', type=str, default="t1_t2_to_flair")
        self.parser.add_argument('--input_modalities', '--input_mods', type=str, default="t1_t2_flair")
        
        self.parser.add_argument('--batch_size', type=int, default=32)
        self.parser.add_argument('--num_workers', type=int, default=4)
        self.parser.add_argument('--num_epochs', type=int, default=200)
        self.parser.add_argument('--num_layers', type=int, default=4)
        self.parser.add_argument('--num_features', type=int, default=32)
        
        self.parser.add_argument('--condition_method', type=str, default="add")
        self.parser.add_argument('--gan_version', type=int, default=3)
        self.parser.add_argument('--dataset_version', type=int, default=3)
        
        self.parser.add_argument('--learning_rate', '--lr', type=float, default=2e-4)
        self.parser.add_argument('--lr_start_epoch', type=int, default=50)
        self.parser.add_argument('--lr_decay', type=float, default=0.95)
        self.parser.add_argument('--lambda_gan_l1', type=float, default=1.0)
        self.parser.add_argument('--lambda_bce', type=float, default=0.1)
        self.parser.add_argument('--lambda_recon', type=float, default=1.0)
        self.parser.add_argument('--lambda_gdl', type=float, default=1.0)
        self.parser.add_argument('--b1', type=float, default=0.5)
        self.parser.add_argument('--b2', type=float, default=0.999)
        self.parser.add_argument('--dropout', type=float, default=0.5)
        
        self.parser.add_argument('--load_model', type=bool, default=False)
        self.parser.add_argument('--save_model', type=bool, default=True)
        
        self.parser.add_argument('--log_interval', type=int, default=5)
        self.parser.add_argument('--checkpoint_interval', type=int, default=5)
        self.parser.add_argument('--save_results_dir_name', '--save_dir_name', type=str, default="t1_t2_flair_cgan_v2_2")
        self.parser.add_argument('--save_checkpoint_dir', type=str, default="/rds/general/user/ql1623/home/datasetGAN/checkpoints")
        self.parser.add_argument('--save_results_dir', type=str, default="/rds/general/user/ql1623/home/datasetGAN/results")

        self.parser.add_argument('--load_results_dir_name', '--load_dir_name', type=str, default="t1_t2_flair_cgan_v2_2") # for test.py
        self.parser.add_argument('--load_epoch', type=int, default=200)
        
    def parse(self):
        self.options = self.parser.parse_args()
        # for attr, value in vars(self.options).items():
        #     setattr(self.options, attr.upper(), value)
        # items = list(vars(self.options).items())
        # for attr, value in items:
        #     setattr(self.options, attr.upper(), value)
        # return self.options
        self.options.gpu_ids = list(map(int, self.options.gpu_ids.split(',')))

        if torch.cuda.is_available() and torch.cuda.device_count() > 1 and self.options.gpu_ids == "0":
            self.options.gpu_ids = list(range(torch.cuda.device_count()))

        self.uppercased_options = argparse.Namespace()
        for attr, value in vars(self.options).items():
            setattr(self.uppercased_options, attr.upper(), value)
        return self.uppercased_options

    def save_options(self, run_id, file_dir, dir_name, train=True):
        if train:
            save_path = os.path.join(file_dir, "chkpt_" + dir_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                
            with open(os.path.join(save_path, "train_opt.txt"), 'a') as file:
                file.write(f"================ Training Options {run_id} ================\n")
                for arg, value in vars(self.uppercased_options).items():
                    file.write(f"{arg}: {value}\n")
            print("Training Options saved to: ", save_path)
                        
        else:
            save_path = os.path.join(file_dir, dir_name + "_test")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                
            with open(os.path.join(save_path, "test_opt.txt"), 'a') as file:
                file.write(f"================ Testing Options {run_id} ================\n")
                for arg, value in vars(self.uppercased_options).items():
                    file.write(f"{arg}: {value}\n")
            print("Testing Options saved to: ", save_path)
    
if __name__ == "__main__":
    import torch.nn as nn
    import model.cgan.generator_unet4_cgan_v2 as models  
    
    options_parser = TrainOptions()
    options = options_parser.parse()
    
    # print(f"Using device: {options.device}")
    print(f"Learning Rate: {options.LEARNING_RATE}")
    print(f"Using device: {options.DEVICE}")
    print(f"Batch size: {options.BATCH_SIZE}")
    # print(f"Batch size: {options.batch_size}")

    options.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    
    # print(f"Using device: {options.device}")
    print(f"Using device: {options.DEVICE}")
    # print(f"Using device: {options.DATA}")
    gen = models.datasetGAN(input_channels=1, output_channels=1, ngf=32, version=3)
    # summary(gen, (2, 128, 128))
    disc = models.Discriminator(in_channels=1, features=[32,64,128,256,512])
    if torch.cuda.device_count() > 1:
        gpu_ids = options.GPU_IDS# set it by counting the gpu instead of waiting for it to be specified
        gen = nn.DataParallel(gen, device_ids=gpu_ids) # DistributedDataParallel?
        disc = nn.DataParallel(disc, device_ids=gpu_ids)

    run_id = "just_trying"
    options_parser.save_options(run_id, "options_try", "try_1", train=True)