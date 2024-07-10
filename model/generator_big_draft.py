import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, act_fn="leaky", norm="batch", use_dropout=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_channels) if norm=="batch" else nn.InstanceNorm2d(output_channels),
            nn.LeakyReLU(0.2, inplace=True) if act_fn=="leaky" else nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_channels) if norm=="batch" else nn.InstanceNorm2d(output_channels),
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
    def __init__(self, input_channels, output_channels, down=True, act_fn="leaky", norm="batch", use_dropout=False):
        super(SamplingBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=4, stride=2, padding=1)   if down
            else nn.ConvTranspose2d(input_channels, output_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(output_channels) if norm=="batch" else nn.InstanceNorm2d(output_channels),
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
        

class ModalityEncodingBlock(nn.Module):
    def __init__(self, input_channels, output_channels, features=[64, 128, 256, 512]):
        super().__init__()
        
        # Encoding part
        self.initial_conv = ConvBlock(input_channels, features[0], act_fn="leaky", norm="batch", use_dropout=False)

        self.initial_down = SamplingBlock(features[0], features[0], down=True, act_fn="leaky", norm="batch", use_dropout=False)
        
        self.down_blocks = nn.ModuleList()
        for feature in features[:-1]:
            self.down_blocks.append(ConvBlock(feature, feature*2, act_fn="leaky", norm="batch", use_dropout=False))
            self.down_blocks.append(SamplingBlock(feature*2, feature*2, down=True, act_fn="leaky", norm="batch", use_dropout=False))
        
        # bottle neck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1], features[-1]*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Decoding part
        self.up_blocks = nn.ModuleList()
        for feature in features[::-1]:
            self.up_blocks.append(SamplingBlock(feature*2, feature, down=False, act_fn="relu", norm="batch", use_dropout=False))
            self.up_blocks.append(ConvBlock(feature*2, feature, act_fn="relu", norm="batch", use_dropout=False))

        self.final_conv = nn.Sequential(
            nn.Conv2d(features[0], output_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        skip_connections = []
        
         # initial conv
        x = self.initial_conv(x)
        skip_connections.append(x)
        x = self.initial_down(x)
        
        # Encoding part
        for i in range(0, len(self.down_blocks), 2):
            x = self.down_blocks[i](x)
            skip_connections.append(x)
            x = self.down_blocks[i + 1](x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoding part
        skip_connections = skip_connections[::-1]
        for i in range(0, len(self.up_blocks), 2):
            x = self.up_blocks[i](x)
            x = torch.cat((x, skip_connections[i // 2]), dim=1)
            x = self.up_blocks[i + 1](x)

        # final conv
        x = self.final_conv(x)

        return x
        
def test():
    x = torch.randn((1, 1, 128, 128))
    model = ModalityEncodingBlock(input_channels=1, output_channels=8, features=[64, 128, 256, 512])
    preds = model(x)
    print(preds.shape)  # Should get back input x shape (1, 1, 128, 128)
    
    # features=[64, 128, 256, 512, 1024]
    # skip_connections = []
    # # initial_conv = nn.Sequential(
    # #     nn.Conv2d(1, features[0], 4, 2, 1),
    # #     nn.LeakyReLU(0.2)
    # # )
    # # x = initial_conv(x)
    # # print(x.shape)
    # # skip_connections.append(x)
    # # print("-------------")
    
    # block = ConvBlock(1, features[0], act_fn="leaky", norm="batch", use_dropout=False)
    # x = block(x)
    # print(x.shape)
    # skip_connections.append(x)
    # block = SamplingBlock(features[0], features[0], down=True, act_fn="leaky", norm="batch", use_dropout=False)
    # x = block(x)
    # print(x.shape)
    # print("------first down-------")

    # block = ConvBlock(features[0], features[1], act_fn="leaky", norm="batch", use_dropout=False)
    # x = block(x)
    # print(x.shape)
    # skip_connections.append(x)
    # block = SamplingBlock(features[1], features[1], down=True, act_fn="leaky", norm="batch", use_dropout=False)
    # x = block(x)
    # print(x.shape)
    # print("------second down-------")

    # block = ConvBlock(features[1], features[2], act_fn="leaky", norm="batch", use_dropout=False)
    # x = block(x)
    # print(x.shape)
    # skip_connections.append(x)
    # block = SamplingBlock(features[2], features[2], down=True, act_fn="leaky", norm="batch", use_dropout=False)
    # x = block(x)
    # print(x.shape)
    # print("------3rd down-------")

    # block = ConvBlock(features[2], features[3], act_fn="leaky", norm="batch", use_dropout=False)
    # x = block(x)
    # print(x.shape)
    # skip_connections.append(x)
    # block = SamplingBlock(features[3], features[3], down=True, act_fn="leaky", norm="batch", use_dropout=False)
    # x = block(x)
    # print(x.shape)
    # print("------4th down-------")

    # # bottle neck, see if use one conv or conv block
    # bottleneck = nn.Sequential(
    #     nn.Conv2d(features[3], features[4], kernel_size=3, stride=1, padding=1),
    #     nn.ReLU(inplace=True)
    # )
    # x = bottleneck(x)
    # print(x.shape)
    # print("------bottleneck-------")


    # # upsampling    
    # skip_connections = skip_connections[::-1]
    # # up then concat
    # block = SamplingBlock(features[4], features[3], down=False, act_fn="relu", norm="batch", use_dropout=False)
    # x = block(x)
    # print(x.shape)
    # # print(skip_connections[0].shape)
    # # print(len(skip_connections))

    # x = torch.cat((x, skip_connections[0]), dim=1)
    # print(x.shape)

    # # convblock 
    # block = ConvBlock(features[3]*2, features[3], act_fn="relu", norm="batch", use_dropout=False) # *2 as we concat in so now channel dims is twice
    # x = block(x)
    # print(x.shape)
    # print("------1st up-------")

    # # up then concat
    # block = SamplingBlock(features[3], features[2], down=False, act_fn="relu", norm="batch", use_dropout=False)
    # x = block(x)
    # print(x.shape)

    # x = torch.cat((x, skip_connections[1]), dim=1)
    # print(x.shape)

    # # convblock
    # block = ConvBlock(features[2]*2, features[2], act_fn="relu", norm="batch", use_dropout=False) # *2 as we concat in so now channel dims is twice
    # x = block(x)
    # print(x.shape)
    # print("------2nd up-------")

    # # up then concat
    # block = SamplingBlock(features[2], features[1], down=False, act_fn="relu", norm="batch", use_dropout=False)
    # x = block(x)
    # print(x.shape)

    # x = torch.cat((x, skip_connections[2]), dim=1)
    # print(x.shape)

    # # convblock
    # block = ConvBlock(features[1]*2, features[1], act_fn="relu", norm="batch", use_dropout=False) # *2 as we concat in so now channel dims is twice
    # x = block(x)
    # print(x.shape)
    # print("------3rd up-------")

    # # up then concat
    # block = SamplingBlock(features[1], features[0], down=False, act_fn="relu", norm="batch", use_dropout=False)
    # x = block(x)
    # print(x.shape)

    # x = torch.cat((x, skip_connections[3]), dim=1)
    # print(x.shape)

    # # convblock
    # block = ConvBlock(features[0]*2, features[0], act_fn="relu", norm="batch", use_dropout=False) # *2 as we concat in so now channel dims is twice
    # x = block(x)
    # print(x.shape)
    # print("------4th up-------")

    # # output layer
    # final_conv = nn.Sequential(
    #     nn.Conv2d(features[0], out_channels=1, kernel_size=3, stride=1, padding=1),
    #     nn.Tanh()
    # )

    # x = final_conv(x)
    # print(x.shape)

    
if __name__ == '__main__':
    test()
    