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
    models_config: str = make_abs_path('./models.yaml')  # portrait animation config
    checkpoint_F: str = make_abs_path('../../pretrained_weights/liveportrait/base_models/appearance_feature_extractor.pth')  # path to checkpoint
    checkpoint_M: str = make_abs_path('../../pretrained_weights/liveportrait/base_models/motion_extractor.pth')  # path to checkpoint
    checkpoint_G: str = make_abs_path('../../pretrained_weights/liveportrait/base_models/spade_generator.pth')  # path to checkpoint
    checkpoint_W: str = make_abs_path('../../pretrained_weights/liveportrait/base_models/warping_module.pth')  # path to checkpoint

    checkpoint_S: str = make_abs_path('../../pretrained_weights/liveportrait/retargeting_models/stitching_retargeting_module.pth')  # path to checkpoint
    flag_use_half_precision: bool = True  # whether to use half precision

    flag_crop_driving_video: bool = False  # whether to crop the driving video, if driving info is a video
    flag_lip_zero: bool = True  # whether let the lip to close state before animation, only take effect when flag_eye_retargeting and flag_lip_retargeting is False
    lip_zero_threshold: float = 0.03

    flag_eye_retargeting: bool = False
    flag_lip_retargeting: bool = False
    flag_stitching: bool = True  # we recommend setting it to True!

    flag_relative: bool = True  # whether to use relative motion
    anchor_frame: int = 0  # set this value if find_best_frame is True

    input_shape: Tuple[int, int] = (256, 256)  # input shape
    output_format: Literal['mp4', 'gif'] = 'mp4'  # output video format
    crf: int = 15  # crf for output video
    output_fps: int = 25 # default output fps

    flag_write_result: bool = True  # whether to write output video
    flag_pasteback: bool = True  # whether to paste-back/stitch the animated face cropping from the face-cropping space to the original image space
    mask_crop: ndarray = cv2.imread(make_abs_path('../utils/resources/mask_template.png'), cv2.IMREAD_COLOR)
    flag_write_gif: bool = False
    size_gif: int = 256
    source_max_dim: int = 1280 # the max dim of height and width of source image
    source_division: int = 2 # make sure the height and width of source image can be divided by this number

    device_id: int = 0
    flag_do_crop: bool = True  # whether to crop the source portrait to the face-cropping space
    flag_do_rot: bool = True  # whether to conduct the rotation when flag_crop_source_image is True
