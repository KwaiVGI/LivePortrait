# coding: utf-8

"""
config for user
"""

import os.path as osp
from dataclasses import dataclass
import tyro
from typing_extensions import Annotated
from .base_config import PrintableConfig, make_abs_path


@dataclass(repr=False)  # use repr from PrintableConfig
class ArgumentConfig(PrintableConfig):
    ########## input arguments ##########
    source_image: Annotated[str, tyro.conf.arg(aliases=["-s"])] = make_abs_path('../../assets/examples/source/s6.jpg')  # path to the source portrait
    driving_info:  Annotated[str, tyro.conf.arg(aliases=["-d"])] = make_abs_path('../../assets/examples/driving/d0.mp4')  # path to driving video or template (.pkl format)
    output_dir: Annotated[str, tyro.conf.arg(aliases=["-o"])] = 'animations/'  # directory to save output video
    #####################################

    ########## inference arguments ##########
    device_id: int = 0
    flag_lip_zero : bool = True # whether let the lip to close state before animation, only take effect when flag_eye_retargeting and flag_lip_retargeting is False
    flag_eye_retargeting: bool = False
    flag_lip_retargeting: bool = False
    flag_stitching: bool = True  # we recommend setting it to True!
    flag_relative: bool = True  # whether to use relative motion
    flag_pasteback: bool = True  # whether to paste-back/stitch the animated face cropping from the face-cropping space to the original image space
    flag_do_crop: bool = True  # whether to crop the source portrait to the face-cropping space
    flag_do_rot: bool = True  # whether to conduct the rotation when flag_do_crop is True
    #########################################

    ########## crop arguments ##########
    dsize: int = 512
    scale: float = 2.3
    vx_ratio: float = 0  # vx ratio
    vy_ratio: float = -0.125  # vy ratio +up, -down
    ####################################

    ########## gradio arguments ##########
    server_port: Annotated[int, tyro.conf.arg(aliases=["-p"])]  = 8890
    share: bool = True
    server_name: str = "0.0.0.0"
