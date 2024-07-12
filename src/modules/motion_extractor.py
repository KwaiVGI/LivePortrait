# coding: utf-8

"""
Motion extractor(M), which directly predicts the canonical keypoints, head pose and expression deformation of the input image
"""

from torch import nn
import torch

from .convnextv2 import convnextv2_tiny
from .util import filter_state_dict

model_dict = {
    'convnextv2_tiny': convnextv2_tiny,
}


class MotionExtractor(nn.Module):
    def __init__(self, **kwargs):
        super(MotionExtractor, self).__init__()

        # default is convnextv2_base
        # backbone = kwargs.get('backbone', 'convnextv2_tiny')
        # self.detector = model_dict.get(backbone)(**kwargs)
        self.detector = convnextv2_tiny(num_kp=21)
        # print("---> %s", kwargs)

    def load_pretrained(self, init_path: str):
        if init_path not in (None, ''):
            state_dict = torch.load(init_path, map_location=lambda storage, loc: storage)['model']
            state_dict = filter_state_dict(state_dict, remove_name='head')
            ret = self.detector.load_state_dict(state_dict, strict=False)
            print(f'Load pretrained model from {init_path}, ret: {ret}')

    def forward(self, x):
        out = self.detector(x)
        return out
