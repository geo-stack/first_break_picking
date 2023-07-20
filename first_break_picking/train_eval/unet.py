# -*- coding: utf-8 -*-
# =============================================================================
# Copyright (C) Les solutions géostack, Inc.
#
# This file is part of the First-Break Picking package.
# Licensed under the terms of the MIT License.
#
# https://github.com/geo-stack/first_break_picking
# =============================================================================

import torch.nn as nn
import torch 
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels:int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3, stride=1,
                      padding=1, bias=False
                      ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=3, stride=1,
                      padding=1, bias=False
                      ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)
    
    
class Down(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels=in_channels,
                       out_channels=out_channels)
        )
    
    def forward(self, x):
        return self.down(x)
    
    
class Up(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 bilinear: bool = False) -> None:
        super().__init__()
        if bilinear:
            # self.up = nn.Upsample(scale_factor=2,
            #                       mode="bilinear",
            #                       align_corners=True)
            raise RuntimeError(
                "Using bilinear is not developped yet"
                )
        else:
            self.up = nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=2, stride=2
            )
        self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels)
        
    def forward(self, x, x1):
        x = self.up(x)
    
        offset_x = x1.size()[3] - x.size()[3]
        offset_y = x1.size()[2] - x.size()[2]
        x = F.pad(x, [
            offset_x//2, offset_x - offset_x//2,
            offset_y//2, offset_y - offset_y//2
        ])
        
        concat_skip = torch.cat((x, x1), dim=1)
        self.conv(concat_skip)
        return x
    
    
class UNet(nn.Module):
    def __init__(self,
                 n_channels: int,
                 out_channels: int,
                 features: list,
                 bilinear: bool
                 ) -> None:
        super().__init__()
        
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        n_features = len(features)
        self.out_channels = out_channels
        
        self.downs.append(
            DoubleConv(
                in_channels=n_channels,
                out_channels=features[0]
                )
        )
        
        for i in range(len(features) - 1):
            self.downs.append(
                Down(
                    in_channels=features[i],
                    out_channels=features[i+1]
                    )
                )
            
            self.ups.append(
                Up(
                    in_channels=features[n_features - 1 - i],
                    out_channels=features[n_features - 2 - i],
                    bilinear=bilinear
                )
            )
            
        self.final = nn.Conv2d(
            in_channels=features[0],
            out_channels=out_channels,
            kernel_size=1
        )
        
    def forward (self, x) :
        skip_connections = []
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            # print(x.shape)
        
        x = skip_connections[-1]
        for i, up in enumerate(self.ups):
            x = up(
                x,
                skip_connections[-i-2]
                )

        out = self.final(x)
        return out


def test(network=False):
    imgs = torch.rand((1, 1, 512, 30))
    net = UNet(n_channels=1, out_channels=2, 
               features=[16, 32, 64, 128],
               bilinear=False
               )
    x_out = net(imgs)
    print(f"\ninput shape: {imgs.shape}, \noutput shape: {x_out.shape}")
    n_parameters = [len(parameters) for parameters in net.parameters()]
    print(f"Number of parameters: {sum(n_parameters)}")
    
    assert imgs[:,0,...].shape == x_out[:,0,...].shape,\
        "FAILED! Size of input and output should be the same, but we got different values"\
        f"input shape: {imgs[:,0,...].shape}, output shape: {x_out[:,0,...].shape}"
    
    print("\n======== PASSED THE TEST ========\n")
    if network == True:
        print(net)
    
if __name__ == "__main__":
    # test(network=True)
    import segmentation_models_pytorch as smp

    model = smp.Unet(
        encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=2,                      # model output channels (number of classes in your dataset)
        decoder_channels =[16, 32, 64, 128][::-1],
        encoder_depth=4
    )
    # print(model)
    