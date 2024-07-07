# coding: utf-8

"""
config dataclass used for inference
"""

import os.path as osp
import cv2
from numpy import ndarray
from dataclasses import dataclass
from typing import Literal, Tuple
from .base_config import PrintableConfig, make_abs_path


@dataclass(repr=False)  # use repr from PrintableConfig
class InferenceConfig(PrintableConfig):
    # MODEL CONFIG, NOT EXPOERTED PARAMS
    models_config: str = make_abs_path('./models.yaml')  # portrait animation config
    checkpoint_F: str = make_abs_path('../../pretrained_weights/liveportrait/base_models/appearance_feature_extractor.pth')  # path to checkpoint of F
    checkpoint_M: str = make_abs_path('../../pretrained_weights/liveportrait/base_models/motion_extractor.pth')  # path to checkpoint pf M
    checkpoint_G: str = make_abs_path('../../pretrained_weights/liveportrait/base_models/spade_generator.pth')  # path to checkpoint of G
    checkpoint_W: str = make_abs_path('../../pretrained_weights/liveportrait/base_models/warping_module.pth')  # path to checkpoint of W
    checkpoint_S: str = make_abs_path('../../pretrained_weights/liveportrait/retargeting_models/stitching_retargeting_module.pth')  # path to checkpoint to S and R_eyes, R_lip

    # EXPOERTED PARAMS
    flag_use_half_precision: bool = True
    flag_crop_driving_video: bool = False
    device_id: int = 0
    flag_lip_zero: bool = True
    flag_eye_retargeting: bool = False
    flag_lip_retargeting: bool = False
    flag_stitching: bool = True
    flag_relative_motion: bool = True
    flag_pasteback: bool = True
    flag_do_crop: bool = True
    flag_do_rot: bool = True

    # NOT EXPOERTED PARAMS
    lip_zero_threshold: float = 0.03 # threshold for flag_lip_zero
    anchor_frame: int = 0 # TO IMPLEMENT

    input_shape: Tuple[int, int] = (256, 256)  # input shape
    output_format: Literal['mp4', 'gif'] = 'mp4'  # output video format
    crf: int = 15  # crf for output video
    output_fps: int = 25 # default output fps

    mask_crop: ndarray = cv2.imread(make_abs_path('../utils/resources/mask_template.png'), cv2.IMREAD_COLOR)
    size_gif: int = 256 # default gif size, TO IMPLEMENT
    source_max_dim: int = 1280 # the max dim of height and width of source image
    source_division: int = 2 # make sure the height and width of source image can be divided by this number
