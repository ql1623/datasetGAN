import torch
import torch.nn as nn
from torchsummary import summary

"""Have Segmentation Network inside GAN, take feature layers at bottleneck instead of right before out_conv of fusion
removed other dataset version as will be using version 4 anyways""" 

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
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.conv(x)
        if self.use_dropout:
            return self.dropout(x)
        else:
            return x

class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_fn="leaky", norm="batch", use_dropout=False):
        """Convolution Block for a Double Convolution operation
        
        Parameters:
            in_channels (int)  : the number of channels from input images.
            out_channels (int) : the number of channels in output images.
            act_fn (str)       : the activation function: leakyReLU | ReLU.
            norm (str)         : the type of normalization: batch | instance.
            use_dropout (bool) : if dropout is used
        """
        super(DoubleConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels) if norm=="batch" else nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True) if act_fn=="leaky" else nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
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
    def __init__(self, in_channels, out_channels, groups, act_fn="leaky", norm="batch", initial=False):
        """Fuse same-level feature layers of different modalities

        Parameters:
            in_channels (int)  : the number of channels from each modality input images.
            out_channels (int) : the number of channels in fused output images.
            groups (int)       : the number of groups to separate data into and perform conv with.
            act_fn (str)       : the activation function: leakyReLU | ReLU.
            norm (str)         : the type of normalization: batch | instance.
            initial (bool)     : if it is the first fusion block (as first will not have previous fused representation to be merge with).
        """
        super(FusionBlock, self).__init__()
        # self.initial = initial
        
        self.initial = initial
        # conv 3d to merge representation of same-level feature layers from both modality
        self.group_conv =  nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(2, 3, 3), padding=(0, 1, 1), groups=groups),
            # nn.Conv3d(in_channels, out_channels, kernel_size=(2 if initial else 2, 3, 3), padding=(0, 1, 1), groups=groups),
            # nn.Conv3d(in_channels, in_channels, kernel_size=(3, 3, 3), padding=(0, 1, 1), groups=groups), for if want like unet in fusion net as well?
            nn.BatchNorm3d(out_channels) if norm=="batch" else nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True) if act_fn=="leaky" else nn.ReLU(inplace=True),
        )
        
    def forward(self, x1, x2): #, condition=None
        # concat in new dim
        # if self.initial: 
        #     if condition is not None:
        #         condition = (condition.unsqueeze(2).unsqueeze(3)).repeat(1, 1, x1.shape[2], x1.shape[3]) # [B, C] --> [B, C=3, H, W]
        #         x = torch.stack((x1, x2, condition), dim=2) # [B, C, H, W] --> [B, C, stack, H, W] = [B, C=256, stack=2+1, H, W]
        # else:
        x = torch.stack((x1, x2), dim=2) # [B, C, H, W] --> [B, C, stack, H, W] = [B, C=256, stack=2, H, W]
        # if condition:  # condition = [B, C=256, stack=3, feat_img_size, feat_img_size]
        #     x = torch.cat((x,condition), dim=2)
        # then swap new dim with channel dim
        # x = x.permute(0, 2, 1, 3, 4).contiguous() # [B, stack=2, C=256, H, W] --> [B, C=256, stack=2+1, H, W], then conv3d on [stack=2+1, H, W]
        # implement the conv3d defined 
        x = self.group_conv(x)
        x = x.squeeze(2)
        return x
    
    
class FusionBlock_with_cond(nn.Module):
    def __init__(self, in_channels, out_channels, groups, act_fn="leaky", norm="batch", is_separate=False):
        """Fuse same-level feature layers of different modalities

        Parameters:
            in_channels (int)  : the number of channels from each modality input images.
            out_channels (int) : the number of channels in fused output images.
            groups (int)       : the number of groups to separate data into and perform conv with.
            act_fn (str)       : the activation function: leakyReLU | ReLU.
            norm (str)         : the type of normalization: batch | instance.
            is_separate (bool) : if the condition features are processed separately before inputting to current fusion
        """
        super(FusionBlock_with_cond, self).__init__()
        # self.initial = initial
        
        self.is_separate = is_separate
        # conv 3d to merge representation of same-level feature layers from both modality
        self.group_conv =  nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(5, 3, 3), padding=(0, 1, 1), groups=groups) if self.is_separate
            else nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=(0, 1, 1), groups=groups),
            nn.BatchNorm3d(out_channels) if norm=="batch" else nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True) if act_fn=="leaky" else nn.ReLU(inplace=True),
        )
        
    def forward(self, x1, x2, condition):
        # concat in new dim
        # condition = [B, C=256, stack=3, feat_img_size, feat_img_size] if is_separate=True
        # condition = [B, C=256, feat_img_size, feat_img_size] if is_separate=False
        if self.is_separate:  # condition = [B, C=256, stack=3, feat_img_size, feat_img_size]
            x = torch.stack((x1, x2), dim=2) # [B, C, H, W] --> [B, C, stack, H, W] = [B, C=256, stack=2, H, W]
            x = torch.cat((x,condition), dim=2)
        else:
            x = torch.stack((x1, x2, condition), dim=2)
        # then swap new dim with channel dim
        # x = x.permute(0, 2, 1, 3, 4).contiguous() # [B, stack=2, C=256, H, W] --> [B, C=256, stack=2+1, H, W], then conv3d on [stack=2+1, H, W]
        # implement the conv3d defined 
        x = self.group_conv(x)
        x = x.squeeze(2)
        return x

class ConditionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, feat_img_size, num_output_comb, has_act_fn=False, is_linear=False): 
        """Transform labels into feature channels to be added to intermediate feautures channels in model
        Note: if embedding layer is chosen, labels need to be pure class labels and not one-hot encoded labels

        Parameters:
            in_channels (int): num channels for labels, ohe_labels = num_modalities, mod_labels = 1
            out_channels (int): num channels in intermediate feautures channels in model
            feat_img_size (int): height and width dims of intermediate feautures channels in model
            num_output_comb (int): number of possible output condition (for embedding layer)
            has_act_fn (bool, optional): if it needs activation function. Defaults to False.
            is_linear (bool, optional): to use linear or embedding layer to transform labels. Defaults to False.
        """
        super(ConditionalBlock,self).__init__()
        self.out_channels = out_channels
        self.feat_img_size = feat_img_size
        self.is_linear = is_linear
        self.has_act_fn = has_act_fn
        
        layers = []
        if self.has_act_fn:
            layers.append(nn.LeakyReLU(0.2))
            
        layers.append(
            nn.Linear(in_channels, out_channels * self.feat_img_size * self.feat_img_size) if self.is_linear
            else nn.Embedding(num_output_comb, out_channels * self.feat_img_size * self.feat_img_size)
        )
        self.cond = nn.Sequential(*layers)
        
    def forward(self, condition):
        # batch_size = condition.shape[0]
        return self.cond(condition.float()).view(condition.shape[0], self.out_channels, self.feat_img_size, self.feat_img_size)
        

class ConditionalBlock_v2(nn.Module):
    def __init__(self, in_channels, out_channels, feat_img_size, is_separate=False): 
        """Transform labels into feature channels to be added to intermediate feautures channels in model
        Note: if embedding layer is chosen, labels need to be pure class labels and not one-hot encoded labels

        Parameters:
            in_channels (int): num channels for labels, ohe_labels = num_modalities, mod_labels = 1
            out_channels (int): num channels in intermediate feautures channels in model
            feat_img_size (int): height and width dims of intermediate feautures channels in model
            num_output_comb (int): number of possible output condition (for embedding layer)
            has_act_fn (bool, optional): if it needs activation function. Defaults to False.
            is_linear (bool, optional): to use linear or embedding layer to transform labels. Defaults to False.
        """
        super(ConditionalBlock_v2,self).__init__()
        self.out_channels = out_channels
        self.feat_img_size = feat_img_size
        # self.has_act_fn = has_act_fn
        self.is_separate = is_separate
        
        # repeat()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, self.out_channels, kernel_size=3, stride=1, padding=1), # out_channels should be 256
            nn.LeakyReLU(0.2, inplace=True)
            )
        
    def forward(self, condition):
        # condition = [B, C=3] 3 as one hot 
        condition = condition.unsqueeze(2).unsqueeze(3).repeat(1, 1, self.feat_img_size, self.feat_img_size) # [B, C=3] --> [B, C=3, feat_img_size, feat_img_size]
        # condition = condition.repeat(1, 1, self.feat_img_size, self.feat_img_size) # [B, C=3, 1, 1] --> [B, C=3, feat_img_size, feat_img_size]
        
        # batch_size = condition.shape[0]
        if self.is_separate: # if is separate, in_channels should be 1
            # [B, C=1, feat_img_size, feat_img_size] --> [B, C=256, feat_img_size, feat_img_size]
            condition_0 = self.conv(condition[:,0,:,:].view(condition.shape[0], 1, self.feat_img_size, self.feat_img_size)) # [B, C=256, feat_img_size, feat_img_size]
            condition_1 = self.conv(condition[:,1,:,:].view(condition.shape[0], 1, self.feat_img_size, self.feat_img_size)) # [B, C=256, feat_img_size, feat_img_size]
            condition_2 = self.conv(condition[:,2,:,:].view(condition.shape[0], 1, self.feat_img_size, self.feat_img_size)) # [B, C=256, feat_img_size, feat_img_size]
            condition = torch.stack((condition_0, condition_1, condition_2), dim=2) # [B, C=256, stack=3, feat_img_size, feat_img_size]
            
        else: # else, in_channels should be 3
            condition = self.conv(condition) # [B, C=3, feat_img_size, feat_img_size] --> [B, C=256, feat_img_size, feat_img_size] to be same as fusion feat to stack 
        return condition

# class SegmentationBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, features):
#         super(SegmentationBlock, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.features = features
        
#         self.conv = nn.Sequential(
#             ConvBlock(self.in_channels, features, act_fn="leaky", norm="batch", use_dropout=False)
#         )
        
#     def forward(self, ):
#         return 

      
# ---------------------------------------------------------------------------------
# =========== define model architecture ============ #
class datasetGAN(nn.Module):
    """ Defining Generator of the model """
    def __init__(self, input_channels, output_channels, ngf, version): # input_channels = 1, output_channel = 1, ngf = 64/32 , features = [16,32,64,128]
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
        self.version = version
        
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
        
        # condition part
        if self.version == 1:
            self.cond = ConditionalBlock(in_channels=3, out_channels=self.features*8, feat_img_size=16, num_output_comb=3, is_linear=True)
            self.fusion0 = FusionBlock(self.features*8, self.features*8, groups=self.features*8, act_fn="relu", norm="batch", initial=True)
        elif self.version == 2:
            self.cond = ConditionalBlock(in_channels=3, out_channels=self.features*8*2, feat_img_size=16, num_output_comb=3, is_linear=True)
            self.fusion0 = FusionBlock(self.features*8, self.features*8, groups=self.features*8, act_fn="relu", norm="batch", initial=True)
        elif self.version == 3:
            self.cond = ConditionalBlock(in_channels=3, out_channels=self.features*8, feat_img_size=16, num_output_comb=3, is_linear=True)
            self.fusion0 = FusionBlock(self.features*8, self.features*8, groups=self.features*8, act_fn="relu", norm="batch", initial=True)
        elif self.version == 4:
            self.cond_v2 = ConditionalBlock_v2(in_channels=1, out_channels=self.features*8, feat_img_size=16, is_separate=True)
            self.fusion0 = FusionBlock_with_cond(self.features*8, self.features*8, groups=self.features*8, act_fn="relu", norm="batch", is_separate=True)
        elif self.version == 5:
            self.cond_v2 = ConditionalBlock_v2(in_channels=3, out_channels=self.features*8, feat_img_size=16, is_separate=False)
            self.fusion0 = FusionBlock_with_cond(self.features*8, self.features*8, groups=self.features*8, act_fn="relu", norm="batch", is_separate=False)
        else:
            raise Exception("version specified is wrong")
        
        # self.fusion0 = FusionBlock(self.features*8, self.features*8, groups=self.features*8, act_fn="relu", norm="batch", initial=True)
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

        # whether to add one more conv so input to seg is lower 
        # self.fusion_conv5 = ConvBlock(self.features, out_channels=16, act_fn="relu", norm="batch", use_dropout=False)
        # -----------
        # self.fusion_final_conv1 = ConvBlock(self.features, self.pre_output_channels, act_fn="relu", norm="batch", use_dropout=False)
        
        # self.fusion_final_conv2 = nn.Sequential(
        #     # nn.Conv2d(self.features, self.features/2, kernel_size=3, stride=1, padding=1),
        #     # nn.ReLU(inplace=True)
        #     nn.Conv2d(self.pre_output_channels, self.output_channels, kernel_size=3, stride=1, padding=1),
        #     nn.Tanh()
        # )
        #  --- or ---
        self.fusion_final_conv = nn.Sequential(
            # nn.Conv2d(self.features, self.features/2, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True)
            nn.Conv2d(self.features, self.input_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        
        # segmentation network
        # 1 initial conv as use feat at bottleneck instead of right before out_conv of fusion
        # version 2
        self.seg_cond = ConditionalBlock(in_channels=3, out_channels=self.features*8*2, feat_img_size=16, num_output_comb=3, is_linear=True)
        self.seg_fusion0 = FusionBlock(self.features*8, self.features*8, groups=self.features*8, act_fn="relu", norm="batch", initial=True)
        self.seg_initial_conv = ConvBlock(in_channels=self.features*8, out_channels=self.features, act_fn="relu", norm="batch", use_dropout=False)
        
        self.seg_conv_1 = DoubleConvBlock(self.features, self.features*2, act_fn="leaky", norm="batch", use_dropout=False)
        self.seg_down_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.seg_conv_2 = DoubleConvBlock(self.features*2, self.features*4, act_fn="leaky", norm="batch", use_dropout=False)
        self.seg_down_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.seg_conv_3 = DoubleConvBlock(self.features*4, self.features*8, act_fn="leaky", norm="batch", use_dropout=False)
        self.seg_down_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.seg_bottleneck = nn.Sequential(
            nn.Conv2d(self.features*8, self.features*8*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.features*8*2),    
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.seg_up_1 = nn.ConvTranspose2d(self.features*8*2, self.features*8, kernel_size=2, stride=2)
        self.seg_upconv_1 = DoubleConvBlock(self.features*8*2, self.features*8, act_fn="leaky", norm="batch", use_dropout=False)

        self.seg_up_2 = nn.ConvTranspose2d(self.features*8, self.features*4, kernel_size=2, stride=2)
        self.seg_upconv_2 = DoubleConvBlock(self.features*4*2, self.features*4, act_fn="leaky", norm="batch", use_dropout=False)

        self.seg_up_3 = nn.ConvTranspose2d(self.features*4, self.features*2, kernel_size=2, stride=2)
        self.seg_upconv_3 = DoubleConvBlock(self.features*2*2, self.features*2, act_fn="leaky", norm="batch", use_dropout=False)

        self.seg_final_conv = nn.Conv2d(self.features*2, self.output_channels, kernel_size=3, stride=1, padding=1)
        # self.seg_final_conv = nn.Conv2d(self.features*2, 1, kernel_size=1, stride=1, padding=1)
        
    def forward(self, inputs, condition):
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
        if self.version == 1:
            condition_feat = self.cond(condition)
            fusion_0 = self.fusion0(bottleneck_x1 + condition_feat, bottleneck_x2 + condition_feat)
            
        elif self.version == 2:
            condition_feat = self.cond(condition)
            fusion_0 = self.fusion0(bottleneck_x1 + condition_feat[:, 0:self.features*8, :, :], bottleneck_x2 + condition_feat[:, self.features*8:, :, :])
        
        elif self.version == 3:
            condition_feat = self.cond(condition)
            fusion_0_pre = self.fusion0(bottleneck_x1, bottleneck_x2)
            fusion_0 = fusion_0_pre + condition_feat
        
        elif self.version == 4:
            condition_feat = self.cond_v2(condition) # [B, C=256, stack=3, feat_img_size, feat_img_size]
            fusion_0 = self.fusion0(bottleneck_x1, bottleneck_x2, condition_feat)
        
        elif self.version == 5:
            condition_feat = self.cond_v2(condition)
            fusion_0 = self.fusion0(bottleneck_x1, bottleneck_x2, condition_feat)
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
        
        # segmentation branch 
        seg_skip_connection = []
        # version 2
        seg_condition_feat = self.seg_cond(condition)
        seg_fusion = self.seg_fusion0(bottleneck_x1 + seg_condition_feat[:, 0:self.features*8, :, :], bottleneck_x2 + seg_condition_feat[:, self.features*8:, :, :])
        
        seg_initial = self.seg_initial_conv(seg_fusion)
        seg_conv_fu1 = self.seg_conv_1(seg_initial)
        seg_skip_connection.append(seg_conv_fu1)
        seg_down_fu1 = self.seg_down_1(seg_conv_fu1)
        
        seg_conv_fu2 = self.seg_conv_2(seg_down_fu1)
        seg_skip_connection.append(seg_conv_fu2)
        seg_down_fu2 = self.seg_down_2(seg_conv_fu2)
        
        seg_conv_fu3 = self.seg_conv_3(seg_down_fu2)
        seg_skip_connection.append(seg_conv_fu3)
        seg_down_fu3 = self.seg_down_3(seg_conv_fu3)
        
        seg_bottleneck_fu = self.seg_bottleneck(seg_down_fu3)
        
        seg_up_fu1 = self.seg_up_1(seg_bottleneck_fu)
        seg_upconv_fu1 = self.seg_upconv_1(torch.cat((seg_up_fu1, seg_skip_connection[2]), dim=1))
        
        seg_up_fu2 = self.seg_up_2(seg_upconv_fu1)
        seg_upconv_fu2 = self.seg_upconv_2(torch.cat((seg_up_fu2, seg_skip_connection[1]), dim=1))
        
        seg_up_fu3 = self.seg_up_3(seg_upconv_fu2)
        seg_upconv_fu3 = self.seg_upconv_3(torch.cat((seg_up_fu3, seg_skip_connection[0]), dim=1))
        
        seg_output = torch.sigmoid(self.seg_final_conv(seg_upconv_fu3))
        
        return out_fusion, out_x1, out_x2, seg_output
 


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

        self.cond = ConditionalBlock(3, out_channels=1, feat_img_size=256, num_output_comb=3, has_act_fn=False, is_linear=True)
        
        self.intitial_conv = nn.Sequential(
            nn.Conv2d(in_channels + 1, features[0], kernel_size=3, stride=2, padding=1),
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

    def forward(self, x, condition):
        condition = self.cond(condition) # [B, C] --> [B, C=1, H, W]
        x = self.intitial_conv(torch.cat((x, condition), 1))
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

    model = datasetGAN(1,1,32, version=2)
    summary(model, [(2, 256, 256),  (3, )]) 
    
    # model = Discriminator(in_channels=1, features=[32,64,128,256,512])
    # summary(model, [(1, 128, 128),  (3, )]) 
    
    
    

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

# current model with fusing at decoder only and unet of 32 feat at start 4 layers of unet with condition at first fusion using add, version 1
# Total params: 14,746,691
# Trainable params: 14,746,691
# Non-trainable params: 0
# Total mult-adds (G): 10.40
# ==========================================================================================
# Input size (MB): 0.13
# Forward/backward pass size (MB): 143.50
# Params size (MB): 56.25
# Estimated Total Size (MB): 199.88

# current model with fusing at decoder only and unet of 32 feat at start 4 layers of unet with condition at first fusion using add, version 2
# Total params: 14,812,227
# Trainable params: 14,812,227
# Non-trainable params: 0
# Total mult-adds (G): 10.40
# ==========================================================================================
# Input size (MB): 0.13
# Forward/backward pass size (MB): 143.62
# Params size (MB): 56.50
# Estimated Total Size (MB): 200.25

# current model with fusing at decoder only and unet of 32 feat at start 4 layers of unet with condition at first fusion using add, version 3
# Total params: 14,746,691
# Trainable params: 14,746,691
# Non-trainable params: 0
# Total mult-adds (G): 10.40
# ==========================================================================================
# Input size (MB): 0.13
# Forward/backward pass size (MB): 143.50
# Params size (MB): 56.25
# Estimated Total Size (MB): 199.88

# current model with fusing at decoder only and unet of 32 feat at start 4 layers of unet with condition at first fusion using concat, version 4
# Total params: 14,690,627
# Trainable params: 14,690,627
# Non-trainable params: 0
# Total mult-adds (G): 10.40
# ==========================================================================================
# Input size (MB): 0.13
# Forward/backward pass size (MB): 143.50
# Params size (MB): 56.04
# Estimated Total Size (MB): 199.67

# current model with fusing at decoder only and unet of 32 feat at start 4 layers of unet with condition at first fusion using concat, version 5
# Total params: 14,690,627
# Trainable params: 14,690,627
# Non-trainable params: 0
# Total mult-adds (G): 10.40
# ==========================================================================================
# Input size (MB): 0.13
# Forward/backward pass size (MB): 143.50
# Params size (MB): 56.04
# Estimated Total Size (MB): 199.67

# current model with fusing at decoder only and unet of 32 feat at start 4 layers of unet with condition at first fusion using add, version 1 with seg

# current model with fusing at decoder only and unet of 32 feat at start 4 layers of unet with condition at first fusion using add, version 2 with seg

# current model with fusing at decoder only and unet of 32 feat at start 4 layers of unet with condition at first fusion using add, version 3 with seg

# current model with fusing at decoder only and unet of 32 feat at start 4 layers of unet with condition at first fusion using concat, version 4 with seg
# Total params: 20,049,988
# Trainable params: 20,049,988
# Non-trainable params: 0
# Total mult-adds (G): 20.48
# ==========================================================================================
# Input size (MB): 0.13
# Forward/backward pass size (MB): 271.62
# Params size (MB): 76.48
# Estimated Total Size (MB): 348.23

# current model with fusing at decoder only and unet of 32 feat at start 4 layers of unet with condition at first fusion using concat, version 5 with seg



# generator_unet4_cgan_v2_seg_v1_2 where seg use cond and fusion at bottleneck then do a conv to get low feat then unet
# (current model with fusing at decoder only and unet of 32 feat at start 4 layers of unet with condition at first fusion using add, version 2 with seg)
# Total params: 21,168,292
# Trainable params: 21,168,292
# Non-trainable params: 0
# Total mult-adds (G): 41.69
# ==========================================================================================
# Input size (MB): 0.50
# Forward/backward pass size (MB): 578.63
# Params size (MB): 80.75
# Estimated Total Size (MB): 659.88