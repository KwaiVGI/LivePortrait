# coding: utf-8

"""
Spade decoder(G) defined in the paper, which input the warped feature to generate the animated image.
"""

import torch
from torch import nn
import torch.nn.functional as F
from .util import SPADEResnetBlock


class SPADEDecoder(nn.Module):
    def __init__(self, upscale=1, max_features=256, block_expansion=64, out_channels=64, num_down_blocks=2):
        for i in range(num_down_blocks):
            input_channels = min(max_features, block_expansion * (2 ** (i + 1)))
        self.upscale = upscale
        super().__init__()
        norm_G = 'spadespectralinstance'
        label_num_channels = input_channels  # 256

        self.fc = nn.Conv2d(input_channels, 2 * input_channels, 3, padding=1)
        self.G_middle_0 = SPADEResnetBlock(2 * input_channels, 2 * input_channels, norm_G, label_num_channels)
        self.G_middle_1 = SPADEResnetBlock(2 * input_channels, 2 * input_channels, norm_G, label_num_channels)
        self.G_middle_2 = SPADEResnetBlock(2 * input_channels, 2 * input_channels, norm_G, label_num_channels)
        self.G_middle_3 = SPADEResnetBlock(2 * input_channels, 2 * input_channels, norm_G, label_num_channels)
        self.G_middle_4 = SPADEResnetBlock(2 * input_channels, 2 * input_channels, norm_G, label_num_channels)
        self.G_middle_5 = SPADEResnetBlock(2 * input_channels, 2 * input_channels, norm_G, label_num_channels)
        self.up_0 = SPADEResnetBlock(2 * input_channels, input_channels, norm_G, label_num_channels)
        self.up_1 = SPADEResnetBlock(input_channels, out_channels, norm_G, label_num_channels)
        self.up = nn.Upsample(scale_factor=2)

        if self.upscale is None or self.upscale <= 1:
            self.conv_img = nn.Conv2d(out_channels, 3, 3, padding=1)
        else:
            self.conv_img = nn.Sequential(
                nn.Conv2d(out_channels, 3 * (2 * 2), kernel_size=3, padding=1),
                nn.PixelShuffle(upscale_factor=2)
            )
        self.patch_nonscriptable_classes()

    def forward(self, feature):
        seg = feature  # Bx256x64x64
        x = self.fc(feature)  # Bx512x64x64
        # print("self.G_middle_0: %s", x.shape, seg.shape)
        x = self.G_middle_0(x, seg)
        # print("self.G_middle_1: %s", x.shape, seg.shape)
        x = self.G_middle_1(x, seg)
        # print("self.G_middle_2: %s", x.shape, seg.shape)
        x = self.G_middle_2(x, seg)
        # print("self.G_middle_3: %s", x.shape, seg.shape)
        x = self.G_middle_3(x, seg)
        # print("self.G_middle_4: %s", x.shape, seg.shape)
        x = self.G_middle_4(x, seg)
        # print("self.G_middle_5: %s", x.shape, seg.shape)
        x = self.G_middle_5(x, seg)

        x = self.up(x)  # Bx512x64x64 -> Bx512x128x128
        # print("self.up_0: %s", x.shape, seg.shape)
        x = self.up_0(x, seg)  # Bx512x128x128 -> Bx256x128x128
        x = self.up(x)  # Bx256x128x128 -> Bx256x256x256
        # print("self.up_1: %s", x.shape, seg.shape)
        x = self.up_1(x, seg)  # Bx256x256x256 -> Bx64x256x256

        x = self.conv_img(F.leaky_relu(x, 2e-1))  # Bx64x256x256 -> Bx3xHxW
        x = torch.sigmoid(x)  # Bx3xHxW

        return x

    def patch_nonscriptable_classes(self):
        in_1 = torch.rand(1, 512, 64, 64)
        in_2 = torch.rand(1, 256, 64, 64)

        in_3 = torch.rand(1, 512, 128, 128)
        in_4 = torch.rand(1, 256, 256, 256)

        self.G_middle_0 = torch.jit.trace(self.G_middle_0.eval(), (in_1, in_2))
        self.G_middle_1 = torch.jit.trace(self.G_middle_1.eval(), (in_1, in_2))
        self.G_middle_2 = torch.jit.trace(self.G_middle_2.eval(), (in_1, in_2))
        self.G_middle_3 = torch.jit.trace(self.G_middle_3.eval(), (in_1, in_2))
        self.G_middle_4 = torch.jit.trace(self.G_middle_4.eval(), (in_1, in_2))
        self.G_middle_5 = torch.jit.trace(self.G_middle_5.eval(), (in_1, in_2))
        self.up_0 = torch.jit.trace(self.up_0.eval(), (in_3, in_2))
        self.up_1 = torch.jit.trace(self.up_1.eval(), (in_4, in_2))
