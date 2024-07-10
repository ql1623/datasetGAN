import torch
import torch.nn as nn
from torchsummary import summary


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_fn="leaky", norm="batch", use_dropout=False):
        """Convolution Block for a Double Convolution operation
        
        Parameters:
            in_channels (int)  : the number of channels from input images.
            out_channels (int) : the number of channels in output images.
            act_fn (str)       : the activation function: leakyReLU | ReLU.
            norm (str)         : the type of normalization: batch | instance.
            use_dropout (bool) : if dropout is used
        """
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels) if norm=="batch" else nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True) if act_fn=="leaky" else nn.ReLU(inplace=True),
            # nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(out_channels) if norm=="batch" else nn.InstanceNorm2d(out_channels),
            # nn.LeakyReLU(0.2, inplace=True) if act_fn=="leaky" else nn.ReLU(inplace=True),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.conv(x)
        if self.use_dropout:
            return self.dropout(x)
        else:
            return x


class SamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act_fn="leaky", norm="batch", use_dropout=False):
        """Convolution Block for upsampling / downsampling
        
        Parameters:
            in_channels (int)  : the number of channels from input images.
            out_channels (int) : the number of channels in output images.
            down (bool)        : if it is a block in downsampling path
            act_fn (str)       : the activation function: leakyReLU | ReLU.
            norm (str)         : the type of normalization: batch | instance.
            use_dropout (bool) : if dropout is used
        """
        super(SamplingBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1) if down
            else nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels) if norm=="batch" else nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True) if act_fn=="leaky" else nn.ReLU(inplace=True),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.conv(x)
        if self.use_dropout:
            return self.dropout(x)
        else:
            return x


class FusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups, act_fn="leaky", norm="batch"):
        """Fuse same-level feature layers of different modalities

        Parameters:
            in_channels (int)  : the number of channels from each modality input images.
            out_channels (int) : the number of channels in fused output images.
            groups (int)       : the number of groups to separate data into and perform conv with.
            initial (bool)     : if it is the first fusion block (as first will not have previous fused representation to be merge with).
            act_fn (str)       : the activation function: leakyReLU | ReLU.
            norm (str)         : the type of normalization: batch | instance.
        """
        super(FusionBlock, self).__init__()
        # self.initial = initial
        
        # conv 3d to merge representation of same-level feature layers from both modality
        self.group_conv =  nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(2, 3, 3), padding=(0, 1, 1), groups=groups),
            # nn.Conv3d(in_channels, in_channels, kernel_size=(3, 3, 3), padding=(0, 1, 1), groups=groups), for if want like unet in fusion net as well?
            nn.BatchNorm3d(out_channels) if norm=="batch" else nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True) if act_fn=="leaky" else nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        # concat in new dim
        x = torch.stack([x1, x2], dim=1) # [B, C, H, W] --> [B, stack, C, H, W] = [B, stack=2, C=64, H, W]
        # then swap new dim with channel dim
        x = x.permute(0, 2, 1, 3, 4).contiguous() # [B, stack=2, C=64, H, W] --> [B, C=64, stack=2, H, W], then conv3d on [stack=2, H, W]
        # implement the conv3d defined 
        x = self.group_conv(x)
        x = x.squeeze(2)
        return x
    

# ---------------------------------------------------------------------------------
# =========== define model architecture ============ #
class datasetGAN(nn.Module):
    """ Defining Generator of the model """
    def __init__(self, input_channels, output_channels, ngf): # input_channels = 1, output_channel = 1, ngf = 64/32 , features = [16,32,64,128]
        """Generator with UNET and Fusion Network 

        Parameters:
            input_channels (int)  : the number of channels from each modality input images.
            output_channels (int) : the number of channels in fused output images.
            ngf (int)       : the number of filter in the first conv layer
        """
        super(datasetGAN,self).__init__()
        
        self.input_channels = input_channels
        self.features = ngf
        self.output_channels = output_channels
        
        # -------- Modality Encoder - Modality 1 -------- 
        # encoding path
        self.down1_mod1 = ConvBlock(in_channels=self.input_channels, out_channels=self.features, act_fn="leaky", norm="batch", use_dropout=False)
        # self.down1_mod1 = ConvBlock(in_channels=self.input_channels*2, out_channels=self.features, act_fn="leaky", norm="batch", use_dropout=False)
        # in_channels=self.input_channels * 2 as adding in labels for cgan
        
        self.pool1_mod1 = SamplingBlock(in_channels=self.features, out_channels=self.features, down=True, act_fn="leaky", norm="batch", use_dropout=False)
        
        self.down2_mod1 = ConvBlock(in_channels=self.features, out_channels=self.features*2, act_fn="leaky", norm="batch", use_dropout=False)
        self.pool2_mod1 = SamplingBlock(in_channels=self.features*2, out_channels=self.features*2, down=True, act_fn="leaky", norm="batch", use_dropout=False)
        
        self.down3_mod1 = ConvBlock(in_channels=self.features*2, out_channels=self.features*4, act_fn="leaky", norm="batch", use_dropout=False)
        self.pool3_mod1 = SamplingBlock(in_channels=self.features*4, out_channels=self.features*4, down=True, act_fn="leaky", norm="batch", use_dropout=False)
        
        self.down4_mod1 = ConvBlock(in_channels=self.features*4, out_channels=self.features*8, act_fn="leaky", norm="batch", use_dropout=False)
        self.pool4_mod1 = SamplingBlock(in_channels=self.features*8, out_channels=self.features*8, down=True, act_fn="leaky", norm="batch", use_dropout=False)
        
        # bottleneck
        self.bottleneck_mod1 = nn.Sequential(
            nn.Conv2d(self.features*8, self.features*8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # decoding path
        self.up1_mod1     = SamplingBlock(self.features*8, self.features*8, down=False, act_fn="relu", norm="batch", use_dropout=False)
        self.upconv1_mod1 = ConvBlock(self.features*8*2, self.features*8, act_fn="relu", norm="batch", use_dropout=False)
        
        self.up2_mod1     = SamplingBlock(self.features*4*2, self.features*4, down=False, act_fn="relu", norm="batch", use_dropout=False)
        self.upconv2_mod1 = ConvBlock(self.features*4*2, self.features*4, act_fn="relu", norm="batch", use_dropout=False)
        
        self.up3_mod1     = SamplingBlock(self.features*2*2, self.features*2, down=False, act_fn="relu", norm="batch", use_dropout=False)
        self.upconv3_mod1 = ConvBlock(self.features*2*2, self.features*2, act_fn="relu", norm="batch", use_dropout=False)
        
        self.up4_mod1     = SamplingBlock(self.features*1*2, self.features*1, down=False, act_fn="relu", norm="batch", use_dropout=False)
        self.upconv4_mod1 = ConvBlock(self.features*1*2, self.features*1, act_fn="relu", norm="batch", use_dropout=False)
        
        # final conv
        self.final_conv_mod1 = nn.Sequential(
            nn.Conv2d(self.features, self.output_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        
        # -------- Modality Encoder - Modality 2 -------- 
        # encoding path
        self.down1_mod2 = ConvBlock(in_channels=self.input_channels, out_channels=self.features, act_fn="leaky", norm="batch", use_dropout=False)
        self.pool1_mod2 = SamplingBlock(in_channels=self.features, out_channels=self.features, down=True, act_fn="leaky", norm="batch", use_dropout=False)
        
        self.down2_mod2 = ConvBlock(in_channels=self.features, out_channels=self.features*2, act_fn="leaky", norm="batch", use_dropout=False)
        self.pool2_mod2 = SamplingBlock(in_channels=self.features*2, out_channels=self.features*2, down=True, act_fn="leaky", norm="batch", use_dropout=False)
        
        self.down3_mod2 = ConvBlock(in_channels=self.features*2, out_channels=self.features*4, act_fn="leaky", norm="batch", use_dropout=False)
        self.pool3_mod2 = SamplingBlock(in_channels=self.features*4, out_channels=self.features*4, down=True, act_fn="leaky", norm="batch", use_dropout=False)
        
        self.down4_mod2 = ConvBlock(in_channels=self.features*4, out_channels=self.features*8, act_fn="leaky", norm="batch", use_dropout=False)
        self.pool4_mod2 = SamplingBlock(in_channels=self.features*8, out_channels=self.features*8, down=True, act_fn="leaky", norm="batch", use_dropout=False)
        
        # bottleneck
        self.bottleneck_mod2 = nn.Sequential(
            nn.Conv2d(self.features*8, self.features*8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # decoding path
        self.up1_mod2     = SamplingBlock(self.features*8, self.features*8, down=False, act_fn="relu", norm="batch", use_dropout=False)
        self.upconv1_mod2 = ConvBlock(self.features*8*2, self.features*8, act_fn="relu", norm="batch", use_dropout=False)
        
        self.up2_mod2     = SamplingBlock(self.features*4*2, self.features*4, down=False, act_fn="relu", norm="batch", use_dropout=False)
        self.upconv2_mod2 = ConvBlock(self.features*4*2, self.features*4, act_fn="relu", norm="batch", use_dropout=False)
        
        self.up3_mod2     = SamplingBlock(self.features*2*2, self.features*2, down=False, act_fn="relu", norm="batch", use_dropout=False)
        self.upconv3_mod2 = ConvBlock(self.features*2*2, self.features*2, act_fn="relu", norm="batch", use_dropout=False)     
           
        self.up4_mod2     = SamplingBlock(self.features*1*2, self.features*1, down=False, act_fn="relu", norm="batch", use_dropout=False)
        self.upconv4_mod2 = ConvBlock(self.features*1*2, self.features*1, act_fn="relu", norm="batch", use_dropout=False)
        
        # final conv
        self.final_conv_mod2 = nn.Sequential(
            nn.Conv2d(self.features, self.output_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
         
        # --------------- fusion network ---------------
        # encoding path       
        # decoding path
        self.fusion0 = FusionBlock(self.features*8, self.features*8, groups=self.features*8, act_fn="relu", norm="batch")
        self.fusion_up0 = SamplingBlock(self.features*8, self.features*8, down=False, act_fn="relu", norm="batch", use_dropout=False)
        
        self.fusion1 = FusionBlock(self.features*8, self.features*8, groups=self.features*8, act_fn="relu", norm="batch")
        self.fusion_conv1 = ConvBlock(self.features*8*2, self.features*8, act_fn="relu", norm="batch", use_dropout=False)
        self.fusion_up1 = SamplingBlock(self.features*8, self.features*4, down=False, act_fn="relu", norm="batch", use_dropout=False)
        
        self.fusion2 = FusionBlock(self.features*4, self.features*4, groups=self.features*4, act_fn="relu", norm="batch")
        self.fusion_conv2 = ConvBlock(self.features*4*2, self.features*4, act_fn="relu", norm="batch", use_dropout=False)
        self.fusion_up2 = SamplingBlock(self.features*4, self.features*2, down=False, act_fn="relu", norm="batch", use_dropout=False)
        
        self.fusion3 = FusionBlock(self.features*2, self.features*2, groups=self.features*2, act_fn="relu", norm="batch")
        self.fusion_conv3 = ConvBlock(self.features*2*2, self.features*2, act_fn="relu", norm="batch", use_dropout=False)
        self.fusion_up3 = SamplingBlock(self.features*2, self.features*1, down=False, act_fn="relu", norm="batch", use_dropout=False)
        
        self.fusion4 = FusionBlock(self.features*1, self.features*1, groups=self.features*1, act_fn="relu", norm="batch")
        self.fusion_conv4 = ConvBlock(self.features*2*1, self.features*1, act_fn="relu", norm="batch", use_dropout=False)

        self.fusion_final_conv = nn.Sequential(
            # nn.Conv2d(self.features, self.features/2, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True)
            nn.Conv2d(self.features, self.input_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        
    def forward(self,inputs):
        # import pdb; pdb.set_trace()
        # inputs.shape = [128,2,128,128] = [batch_size*num_patches, num_modality, H, W]   

        x1 = inputs[:,0:1,:,:] # modality 1
        x2 = inputs[:,1:2,:,:] # modality 2
        
        skip_connections_1 = []
        skip_connections_2 = []
        # feat_down_fusion_1 = []
        # feat_down_fusion_2 = []
        # feat_up_fusion_1 = []
        # feat_up_fusion_2 = []
        
        # ====== # ====== Encoding path ====== # ====== #
        # ----- encoding path - Modality 1 -----
        down1_x1 = self.down1_mod1(x1)
        skip_connections_1.append(down1_x1)
        pool1_x1 = self.pool1_mod1(down1_x1)
        # feat_down_fusion_1.append(pool1_x1)
        
        down2_x1 = self.down2_mod1(pool1_x1)
        skip_connections_1.append(down2_x1)
        pool2_x1 = self.pool2_mod1(down2_x1)
        # feat_down_fusion_1.append(pool2_x1)
        
        down3_x1 = self.down3_mod1(pool2_x1)
        skip_connections_1.append(down3_x1)
        pool3_x1 = self.pool3_mod1(down3_x1)
        # feat_down_fusion_1.append(pool3_x1)
        
        down4_x1 = self.down4_mod1(pool3_x1)
        skip_connections_1.append(down4_x1)
        pool4_x1 = self.pool4_mod1(down4_x1)
        # feat_down_fusion_1.append(pool4_x1)
        bottleneck_x1 = self.bottleneck_mod1(pool4_x1)
        
        # bottleneck_x1 = self.bottleneck_mod1(pool3_x1)
        
        # ----- encoding path - Modality 2 -----
        down1_x2 = self.down1_mod2(x2)
        skip_connections_2.append(down1_x2)
        pool1_x2 = self.pool1_mod2(down1_x2)
        # feat_down_fusion_2.append(pool1_x2)
        
        down2_x2 = self.down2_mod2(pool1_x2)
        skip_connections_2.append(down2_x2)
        pool2_x2 = self.pool2_mod2(down2_x2)
        # feat_down_fusion_2.append(pool2_x2)
        
        down3_x2 = self.down3_mod2(pool2_x2)
        skip_connections_2.append(down3_x2)
        pool3_x2 = self.pool3_mod2(down3_x2)
        # feat_down_fusion_2.append(pool3_x2)
        
        down4_x2 = self.down4_mod2(pool3_x2)
        skip_connections_2.append(down4_x2)
        pool4_x2 = self.pool4_mod2(down4_x2)
        # feat_down_fusion_2.append(pool4_x2)
        bottleneck_x2 = self.bottleneck_mod2(pool4_x2)
        
        # bottleneck_x2 = self.bottleneck_mod2(pool3_x2)
        
        # ====== # ====== Decoding path ====== # ====== #
        # ----- Decoding part - Modality 1 -----
        # skip_connections_1_rev = skip_connections_1[::-1]
        up1_x1 = self.up1_mod1(bottleneck_x1)
        upconv1_x1 = self.upconv1_mod1(torch.cat((up1_x1, skip_connections_1[3]),dim=1))
        
        up2_x1 = self.up2_mod1(upconv1_x1)
        upconv2_x1 = self.upconv2_mod1(torch.cat((up2_x1, skip_connections_1[2]),dim=1))
        
        up3_x1 = self.up3_mod1(upconv2_x1)
        upconv3_x1 = self.upconv3_mod1(torch.cat((up3_x1, skip_connections_1[1]),dim=1))
        
        up4_x1 = self.up4_mod1(upconv3_x1)
        upconv4_x1 = self.upconv4_mod1(torch.cat((up4_x1, skip_connections_1[0]),dim=1))
        
        out_x1 = self.final_conv_mod1(upconv4_x1)
        
        # out_x1 = self.final_conv_mod1(upconv3_x1)
        
        # ----- Decoding part - Modality 2 -----
        # skip_connections_2_rev = skip_connections_2[::-1]
        up1_x2 = self.up1_mod2(bottleneck_x2)
        upconv1_x2 = self.upconv1_mod2(torch.cat((up1_x2, skip_connections_2[3]),dim=1))
        
        up2_x2 = self.up2_mod2(upconv1_x2)
        upconv2_x2 = self.upconv2_mod2(torch.cat((up2_x2, skip_connections_2[2]),dim=1))
        
        up3_x2 = self.up3_mod2(upconv2_x2)
        upconv3_x2 = self.upconv3_mod2(torch.cat((up3_x2, skip_connections_2[1]),dim=1))
        
        up4_x2 = self.up4_mod2(upconv3_x2)
        upconv4_x2 = self.upconv4_mod2(torch.cat((up4_x2, skip_connections_2[0]),dim=1))
        
        out_x2 = self.final_conv_mod2(upconv4_x2)
        # out_x2 = self.final_conv_mod2(upconv3_x2)
        
        # ----- fusion part (for decoding path) -----
        fusion_0 = self.fusion0(bottleneck_x1, bottleneck_x2)
        fusion_upsamp0 = self.fusion_up0(fusion_0)
        
        fusion_1 = self.fusion1(upconv1_x1, upconv1_x2)
        fusion_merge1 = self.fusion_conv1(torch.cat((fusion_1,fusion_upsamp0),dim=1))
        fusion_upsamp1 = self.fusion_up1(fusion_merge1)
        
        fusion_2 = self.fusion2(upconv2_x1, upconv2_x2)
        fusion_merge2 = self.fusion_conv2(torch.cat((fusion_2,fusion_upsamp1),dim=1))
        fusion_upsamp2 = self.fusion_up2(fusion_merge2)
        
        fusion_3 = self.fusion3(upconv3_x1, upconv3_x2)
        fusion_merge3 = self.fusion_conv3(torch.cat((fusion_3,fusion_upsamp2),dim=1))
        fusion_upsamp3 = self.fusion_up3(fusion_merge3)
        
        fusion_4 = self.fusion4(upconv4_x1, upconv4_x2)
        fusion_merge4 = self.fusion_conv4(torch.cat((fusion_4,fusion_upsamp3),dim=1))
        
        out_fusion = self.fusion_final_conv(fusion_merge4)
        # out_fusion = self.fusion_final_conv(fusion_merge3)
        
        return out_fusion, out_x1, out_x2
 


class DownConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, act_fn="leaky", norm="batch"):
        """Convolution Block for a Single Convolution that downsamples at the same time
        
        Parameters:
            in_channels (int)  : the number of channels from input images.
            out_channels (int) : the number of channels in output images.
            kernel_size (int)  : the kernel size of filter used
            stride (int)  : the stride used for convolving
            padding (int) : the padding applied to input images
            act_fn (str)       : the activation function: leakyReLU | ReLU.
            norm (str)         : the type of normalization: batch | instance.    
        """
        super(DownConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels) if norm=="batch" else nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True) if act_fn=="leaky" else nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, features=[32,64,128,256,512]):
        """ Discriminator of the model

        Parameters:
            in_channels (int) : the number of channels from input images. Defaults to 1.
            features (int list) : list of features in each layer of the disc. Defaults to [32,64,128,256,512].
        """
        super(Discriminator, self).__init__()

        self.intitial_conv = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=3, stride=2, padding=1),
            # nn.Conv2d(in_channels*2, features[0], kernel_size=3, stride=2, padding=1), # if cond. GAN
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        layers = []
        for feat in features[:-1]:
            layers.append(
                DownConvBlock(feat, feat*2, kernel_size=3, stride=2, padding=1, act_fn="leaky", norm="batch")
            )
            
        layers.append(nn.Conv2d(features[-1], in_channels, kernel_size=3, stride=1, padding=1)) # should be patchGAN of final out 8x8
        # layers.append(nn.Conv2d(features[-1], in_channels, kernel_size=3, stride=1, padding=1)) # or fully connected layer
        
        self.disc = nn.Sequential(*layers)

    def forward(self, x):
        x = self.intitial_conv(x)
        return self.disc(x)    

def test():
    # x_gen = torch.randn((1,2,128,128))
    # x_disc = torch.randn((1,1,128,128))
    # disc = Discriminator(in_channels=1, features=[32,64,128,256,512])
    # ngf = 16
    # gen = datasetGAN(1,1,ngf)
    # preds_disc = disc(x_disc)
    # preds_gen, pred_x1, pred_x2 = gen(x_gen)
    # print(preds_gen.shape)
    # print(pred_x1.shape)
    # print(pred_x2.shape)
    # print(preds_disc.shape)

    model = datasetGAN(1,1,32)
    summary(model, (2, 128, 128)) 
    

if __name__ == "__main__":
    test()

# big previous model with hi-net but with unet and group conv 
# Total params: 19,716,579
# Trainable params: 19,716,579
# Non-trainable params: 0
# Total mult-adds (G): 9.27
# ==========================================================================================
# Input size (MB): 0.12
# Forward/backward pass size (MB): 129.38
# Params size (MB): 75.21
# Estimated Total Size (MB): 204.71
  
# current model with fusing at decoder only and unet of 16 feat at start 3 layers of unet
# model = datasetGAN(1,1,16)
# summary(model, (2, 128, 128)) 
# Total params: 909,539
# Trainable params: 909,539
# Non-trainable params: 0
# Total mult-adds (G): 1.90
# ==========================================================================================
# Input size (MB): 0.12
# Forward/backward pass size (MB): 67.38
# Params size (MB): 3.47
# Estimated Total Size (MB): 70.97

# current model with fusing at decoder only and unet of 32 feat at start 3 layers of unet
# model = datasetGAN(1,1,32)
# summary(model, (2, 128, 128)) 
# Total params: 3,622,339 ---------> !!!!!!!!!!!!!!!!!
# Trainable params: 3,622,339
# Non-trainable params: 0
# Total mult-adds (G): 7.57
# ==========================================================================================
# Input size (MB): 0.12
# Forward/backward pass size (MB): 134.38
# Params size (MB): 13.82
# Estimated Total Size (MB): 148.32

# current model with fusing at decoder only and unet of 16 feat at start 4 layers of unet
# model = datasetGAN(1,1,16)
# summary(model, (2, 128, 128)) 
# Total params: 3,678,243
# Trainable params: 3,678,243
# Non-trainable params: 0
# Total mult-adds (G): 2.61
# ==========================================================================================
# Input size (MB): 0.12
# Forward/backward pass size (MB): 71.88
# Params size (MB): 14.03
# Estimated Total Size (MB): 86.03

# current model with fusing at decoder only and unet of 32 feat at start 4 layers of unet
# model = datasetGAN(1,1,32)
# summary(model, (2, 128, 128)) 
# Total params: 14,681,155
# Trainable params: 14,681,155
# Non-trainable params: 0
# Total mult-adds (G): 10.40
# ==========================================================================================
# Input size (MB): 0.12
# Forward/backward pass size (MB): 143.38
# Params size (MB): 56.00
# Estimated Total Size (MB): 199.50