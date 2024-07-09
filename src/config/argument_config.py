# coding: utf-8

"""
All configs for user
"""

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

    ########## inference arguments ##########
    flag_use_half_precision: bool = True  # whether to use half precision (FP16). If black boxes appear, it might be due to GPU incompatibility; set to False.
    flag_crop_driving_video: bool = False  # whether to crop the driving video, if the given driving info is a video
    device_id: int = 0 # gpu device id
    flag_force_cpu: bool = False # force cpu inference, WIP
    flag_lip_zero : bool = True # whether let the lip to close state before animation, only take effect when flag_eye_retargeting and flag_lip_retargeting is False
    flag_eye_retargeting: bool = False # not recommend to be True, WIP
    flag_lip_retargeting: bool = False # not recommend to be True, WIP
    flag_stitching: bool = True  # recommend to True if head movement is small, False if head movement is large
    flag_relative_motion: bool = True  # whether to use relative motion
    flag_pasteback: bool = True  # whether to paste-back/stitch the animated face cropping from the face-cropping space to the original image space
    flag_do_crop: bool = True  # whether to crop the source portrait to the face-cropping space
    flag_do_rot: bool = True  # whether to conduct the rotation when flag_do_crop is True

    ########## crop arguments ##########
    scale: float = 2.5 # the ratio of face area is smaller if scale is larger
    vx_ratio: float = 0  # the ratio to move the face to left or right in cropping space
    vy_ratio: float = -0.125  # the ratio to move the face to up or down in cropping space

    ########## gradio arguments ##########
    server_port: Annotated[int, tyro.conf.arg(aliases=["-p"])]  = 8890 # port for gradio server
    share: bool = False # whether to share the server to public
    server_name: str = "0.0.0.0" # server name
