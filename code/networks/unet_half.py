# not used

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np




class UNet3D(nn.Module):
    def __init__(self):
        super(UNet3D, self).__init__()
        
        self.inc = inconv()  #1->32->64

        self.down1 = down(32,64) #MP -> double conv
        self.down2 = down(64,128)
        self.down3 = down(128,256)

        self.up1 = up(256,128)
        self.up2 = up(128,64)
        self.up3 = up(64,32)

        self.outc = outconv()  #64->1
        
        
    def forward(self, x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        #print(f"x1 {x1.shape}\n"
        #f"x2 {x2.shape}\n"
        #f"x3 {x3.shape}\n"
        #f"x4 {x4.shape}\n")
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        return torch.sigmoid(x)


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm3d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self):
        super(inconv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool3d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        self.up_ = nn.ConvTranspose3d(in_ch, in_ch, 2, stride=2)
        self.conv = double_conv(in_ch+out_ch, out_ch)

    def forward(self, x1, x2):
        #print(f"before up: x1: {x1.shape}")
        x1 = self.up_(x1)
        #print(f"after up: x1: {x1.shape}")
        #print(f"x2: {x2.shape}")
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        #print(f"finished up: {x.shape}")
        return x


class outconv(nn.Module):
    def __init__(self):
        super(outconv, self).__init__()
        self.conv = nn.Conv3d(32, 2, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
