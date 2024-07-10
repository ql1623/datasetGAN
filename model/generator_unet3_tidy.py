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
            act_fn (str)       : the activation function: leakyReLU | ReLU.
            norm (str)         : the type of normalization: batch | instance.
        """
        super(FusionBlock, self).__init__()
        
        self.group_conv =  nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(2, 3, 3), padding=(0, 1, 1), groups=groups),
            nn.BatchNorm3d(out_channels) if norm=="batch" else nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True) if act_fn=="leaky" else nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x = torch.stack([x1, x2], dim=1)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.group_conv(x)
        x = x.squeeze(2)
        return x


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, act_fn="leaky", norm="batch"):
        super(UNetBlock, self).__init__()
        self.num_blocks = num_blocks
        
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        # self.pool_blocks = nn.ModuleList()
        # self.up_samp_blocks = nn.ModuleList()
        current_channels = in_channels
        
        for i in range(num_blocks):
            next_channels = out_channels * (2 ** i)
            self.down_blocks.append(ConvBlock(current_channels, next_channels, act_fn, norm))
            self.down_blocks.append(SamplingBlock(next_channels, next_channels, down=True, act_fn=act_fn, norm=norm))
            current_channels = next_channels
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(current_channels, current_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        for i in range(num_blocks):
            if i==0:
                self.up_blocks.append(SamplingBlock(current_channels, current_channels, down=False, act_fn="relu", norm=norm))    
            else: 
                self.up_blocks.append(SamplingBlock(current_channels * 2, current_channels, down=False, act_fn="relu", norm=norm)) 
            self.up_blocks.append(ConvBlock(current_channels * 2, current_channels, act_fn="relu", norm=norm))
            current_channels = current_channels // 2

        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        skip_connections = []
        for index in range(0, len(self.down_blocks), 2):
            x = self.down_blocks[index](x) # conv first
            skip_connections.append(x)
            x = self.down_blocks[index+1](x) # conv2d with stride 2 to down

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for index in range(0, len(self.up_blocks), 2):
            x = self.up_blocks[index](x) # upsampling first 
            skip_feat = skip_connections[index//2]
            x = torch.cat((x, skip_feat), dim=1) # concat
            x = self.up_blocks[index+1](x) # then conv

        x = self.final_conv(x)
        return x


class datasetGAN(nn.Module):
    """ Defining Generator of the model """
    def __init__(self, input_channels, output_channels, ngf, num_blocks): 
        """Generator with UNET and Fusion Network 

        Parameters:
            input_channels (int)  : the number of channels from each modality input images.
            output_channels (int) : the number of channels in fused output images.
            ngf (int)       : the number of filters in the first conv layer
            num_blocks (int): the number of UNet blocks in encoder and decoder paths
        """
        super(datasetGAN, self).__init__()
        
        self.input_channels = input_channels
        self.features = ngf
        self.output_channels = output_channels
        
        self.unet_mod1 = UNetBlock(input_channels, ngf, num_blocks)
        self.unet_mod2 = UNetBlock(input_channels, ngf, num_blocks)
        
        self.fusion_blocks = nn.ModuleList()
        
        # the first part of fusion network
        self.fusion_blocks.append(
            FusionBlock(self.features*(2**(num_blocks-1)), self.features*(2**(num_blocks-1)), groups=self.features*(2**(num_blocks-1)), act_fn="relu", norm="batch")
        )
        self.fusion_blocks.append(
            SamplingBlock(self.features*(2**(num_blocks-1)), self.features*(2**(num_blocks-1)), down=False, act_fn="relu", norm="batch", use_dropout=False)
        )
        
        # the middle part of fusion network
        for i in reversed(range(1, num_blocks)):
            in_features = self.features*(2 ** i)
            out_features = self.features * (2 ** (i - 1))
            self.fusion_blocks.append(FusionBlock(in_features, in_features, groups=in_features, act_fn="relu", norm="batch"))
            self.fusion_blocks.append(ConvBlock(in_features * 2, in_features, act_fn="relu", norm="batch", use_dropout=False))
            self.fusion_blocks.append(SamplingBlock(in_features, out_features, down=False, act_fn="relu", norm="batch", use_dropout=False))

        # the last part of fusion network
        self.fusion_blocks.append(FusionBlock(self.features, self.features, groups=self.features, act_fn="relu", norm="batch"))
        self.fusion_blocks.append(ConvBlock(self.features * 2, self.features, act_fn="relu", norm="batch", use_dropout=False))

        
        self.final_conv = nn.Sequential(
            nn.Conv2d(self.features, self.input_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        
    def forward(self, inputs):
        x1 = inputs[:, 0:1, :, :]
        x2 = inputs[:, 1:2, :, :]
        
        out_mod1 = self.unet_mod1(x1)
        out_mod2 = self.unet_mod2(x2)
        
        prev_fusion = None
        for i in range(0, len(self.fusion_blocks), 3):
            if prev_fusion is None:
                prev_fusion = self.fusion_blocks[i](out_mod1, out_mod2)
                prev_fusion = self.fusion_blocks[i + 1](torch.cat([prev_fusion, prev_fusion], dim=1))
            else:
                prev_fusion = self.fusion_blocks[i](out_mod1, out_mod2)
                prev_fusion = self.fusion_blocks[i + 1](torch.cat([prev_fusion, prev_fusion], dim=1))
                if i + 2 < len(self.fusion_blocks):
                    prev_fusion = self.fusion_blocks[i + 2](prev_fusion)
        
        out_fusion = self.final_conv(prev_fusion)
        
        return out_fusion, out_mod1, out_mod2


# For testing and model summary
if __name__ == "__main__":
    model = datasetGAN(input_channels=1, output_channels=1, ngf=16, num_blocks=3)
    summary(model, input_size=(2, 128, 128))
    # x = torch.randn((1,1,128,128))
    # model = UNetBlock(1,1,3)
    # pred = model(x)
    # print(pred.shape)
