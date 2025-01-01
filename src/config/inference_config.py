# coding: utf-8

"""
config dataclass used for inference
"""

import cv2
from numpy import ndarray
import pickle as pkl
from dataclasses import dataclass, field
from typing import Literal, Tuple
from .base_config import PrintableConfig, make_abs_path

def load_lip_array():
    with open(make_abs_path('../utils/resources/lip_array.pkl'), 'rb') as f:
        return pkl.load(f)

@dataclass(repr=False)  # use repr from PrintableConfig
class InferenceConfig(PrintableConfig):
    # HUMAN MODEL CONFIG, NOT EXPORTED PARAMS
    models_config: str = make_abs_path('./models.yaml')  # portrait animation config
    checkpoint_F: str = make_abs_path('../../pretrained_weights/liveportrait/base_models/appearance_feature_extractor.pth')  # path to checkpoint of F
    checkpoint_M: str = make_abs_path('../../pretrained_weights/liveportrait/base_models/motion_extractor.pth')  # path to checkpoint pf M
    checkpoint_G: str = make_abs_path('../../pretrained_weights/liveportrait/base_models/spade_generator.pth')  # path to checkpoint of G
    checkpoint_W: str = make_abs_path('../../pretrained_weights/liveportrait/base_models/warping_module.pth')  # path to checkpoint of W
    checkpoint_S: str = make_abs_path('../../pretrained_weights/liveportrait/retargeting_models/stitching_retargeting_module.pth')  # path to checkpoint to S and R_eyes, R_lip

    # ANIMAL MODEL CONFIG, NOT EXPORTED PARAMS
    # version_animals = "" # old version
    version_animals = "_v1.1" # new (v1.1) version
    checkpoint_F_animal: str = make_abs_path(f'../../pretrained_weights/liveportrait_animals/base_models{version_animals}/appearance_feature_extractor.pth')  # path to checkpoint of F
    checkpoint_M_animal: str = make_abs_path(f'../../pretrained_weights/liveportrait_animals/base_models{version_animals}/motion_extractor.pth')  # path to checkpoint pf M
    checkpoint_G_animal: str = make_abs_path(f'../../pretrained_weights/liveportrait_animals/base_models{version_animals}/spade_generator.pth')  # path to checkpoint of G
    checkpoint_W_animal: str = make_abs_path(f'../../pretrained_weights/liveportrait_animals/base_models{version_animals}/warping_module.pth')  # path to checkpoint of W
    checkpoint_S_animal: str = make_abs_path('../../pretrained_weights/liveportrait/retargeting_models/stitching_retargeting_module.pth')  # path to checkpoint to S and R_eyes, R_lip, NOTE: use human temporarily!

    # EXPORTED PARAMS
    flag_use_half_precision: bool = True
    flag_crop_driving_video: bool = False
    device_id: int = 0
    flag_normalize_lip: bool = True
    flag_source_video_eye_retargeting: bool = False
    flag_eye_retargeting: bool = False
    flag_lip_retargeting: bool = False
    flag_stitching: bool = True
    flag_relative_motion: bool = True
    flag_pasteback: bool = True
    flag_do_crop: bool = True
    flag_do_rot: bool = True
    flag_force_cpu: bool = False
    flag_do_torch_compile: bool = False
    driving_option: str = "pose-friendly" # "expression-friendly" or "pose-friendly"
    driving_multiplier: float = 1.0
    driving_smooth_observation_variance: float = 3e-7 # smooth strength scalar for the animated video when the input is a source video, the larger the number, the smoother the animated video; too much smoothness would result in loss of motion accuracy
    source_max_dim: int = 1280 # the max dim of height and width of source image or video
    source_division: int = 2 # make sure the height and width of source image or video can be divided by this number
    animation_region: Literal["exp", "pose", "lip", "eyes", "all"] = "all" # the region where the animation was performed, "exp" means the expression, "pose" means the head pose

    # NOT EXPORTED PARAMS
    lip_normalize_threshold: float = 0.03 # threshold for flag_normalize_lip
    source_video_eye_retargeting_threshold: float = 0.18 # threshold for eyes retargeting if the input is a source video
    anchor_frame: int = 0 # TO IMPLEMENT

    input_shape: Tuple[int, int] = (256, 256)  # input shape
    output_format: Literal['mp4', 'gif'] = 'mp4'  # output video format
    crf: int = 15  # crf for output video
    output_fps: int = 25 # default output fps

    mask_crop: ndarray = field(default_factory=lambda: cv2.imread(make_abs_path('../utils/resources/mask_template.png'), cv2.IMREAD_COLOR))
    lip_array: ndarray = field(default_factory=load_lip_array)
    size_gif: int = 256 # default gif size, TO IMPLEMENT
