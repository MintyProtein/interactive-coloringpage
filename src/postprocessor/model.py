import os
from omegaconf import OmegaConf
import numpy as np
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from patchify import patchify, unpatchify

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same', padding_mode='replicate'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding='same', padding_mode='replicate'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, kernel_size)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, kernel_size)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNetPostprocessor(nn.Module):
    def __init__(self, in_channels, out_channels, width, depth, kernel_size, patch_size, device):
        super(UNetPostprocessor, self).__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.inc = DoubleConv(in_channels, width, kernel_size)
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.device = device
        c = width
        for i in range(depth):
            self.downs.append(Down(c, c*2, kernel_size))
            c = c * 2
        self.downs.append(Down(c, c, kernel_size))
        for j in range(depth):
            self.ups.append(Up(c*2, c // 2, kernel_size))
            c = c // 2
        self.ups.append(Up(c*2, c, kernel_size))
        self.outc = OutConv(c, out_channels)
        assert c == width
            
    def forward(self, x):
        assert x.shape[-1] == x.shape[-2] == self.patch_size
        xs = [self.inc(x)]
        for down in self.downs:
            xs.append(down(xs[-1]))
        x = xs.pop()
        for up in self.ups:
            x = up(x, xs.pop())
        return self.outc(x)
    
    @staticmethod
    def load_from_config(config_path, device=torch.device('cuda'), load_checkpoint=True):
        config = OmegaConf.load(config_path)
        model = UNetPostprocessor(in_channels=config.in_channels,
                                  out_channels=config.out_channels,
                                  width=config.model_width,
                                  depth=config.model_depth,
                                  kernel_size=config.kernel_size,
                                  device=device)
        
        if load_checkpoint:
            model.load_state_dict(torch.load(config.model_checkpoint + "/postprocessor.pt"))
        
        model.to(device)
        return model
        
    
    @torch.inference_mode
    def inference(self, img):
        i_h, i_w = img.shape
        patches = patchify(img, (self.patch_size, self.patch_size), self.patch_size)
        n_h, n_w, p_h , p_w = patches.shape
        patches = einops.rearrange(patches, 'nh nw ph pw -> (nh nw) 1 ph pw')
        
    
        x = torch.FloatTensor(patches) / 255
        pred = self.forward(x.to(self.device))
        pred = einops.rearrange(pred, '(nh nw) 1 ph pw -> nh nw ph pw', nh=n_h, nw=n_w)

        out_img = unpatchify(pred.cpu().numpy(), imsize=(i_h, i_w))
        out_img = np.clip(out_img, 0, 1) * 255
        return out_img
    
    