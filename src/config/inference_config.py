# coding: utf-8

"""
config dataclass used for inference
"""

import os.path as osp
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

    flag_lip_zero: bool = True  # whether let the lip to close state before animation, only take effect when flag_eye_retargeting and flag_lip_retargeting is False
    lip_zero_threshold: float = 0.03

    flag_eye_retargeting: bool = False
    flag_lip_retargeting: bool = False
    flag_stitching: bool = True  # we recommend setting it to True!

    flag_relative: bool = True  # whether to use relative pose
    anchor_frame: int = 0  # set this value if find_best_frame is True

    input_shape: Tuple[int, int] = (256, 256)  # input shape
    output_format: Literal['mp4', 'gif'] = 'mp4'  # output video format
    output_fps: int = 30  # fps for output video
    crf: int = 15  # crf for output video

    flag_write_result: bool = True  # whether to write output video
    flag_pasteback: bool = True  # whether to paste-back/stitch the animated face cropping from the face-cropping space to the original image space
    mask_crop = None
    flag_write_gif: bool = False
    size_gif: int = 256
    ref_max_shape: int = 1280
    ref_shape_n: int = 2

    device_id: int = 0
    flag_do_crop: bool = False  # whether to crop the reference portrait to the face-cropping space
    flag_do_rot: bool = True  # whether to conduct the rotation when flag_do_crop is True
