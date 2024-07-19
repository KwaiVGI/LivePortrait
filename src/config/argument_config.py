# coding: utf-8

"""
All configs for user
"""

from dataclasses import dataclass
import tyro
from typing_extensions import Annotated
from typing import Optional
from .base_config import PrintableConfig, make_abs_path


@dataclass(repr=False)  # use repr from PrintableConfig
class ArgumentConfig(PrintableConfig):
    ########## input arguments ##########
    source: Annotated[str, tyro.conf.arg(aliases=["-s"])] = make_abs_path('../../assets/examples/source/s0.jpg')  # path to the source portrait or video
    driving:  Annotated[str, tyro.conf.arg(aliases=["-d"])] = make_abs_path('../../assets/examples/driving/d0.mp4')  # path to driving video or template (.pkl format)
    output_dir: Annotated[str, tyro.conf.arg(aliases=["-o"])] = 'animations/'  # directory to save output video

    ########## inference arguments ##########
    flag_use_half_precision: bool = True  # whether to use half precision (FP16). If black boxes appear, it might be due to GPU incompatibility; set to False.
    flag_crop_driving_video: bool = False  # whether to crop the driving video, if the given driving info is a video
    device_id: int = 0  # gpu device id
    flag_force_cpu: bool = False  # force cpu inference, WIP!
    flag_normalize_lip: bool = True  # whether to let the lip to close state before animation, only take effect when flag_eye_retargeting and flag_lip_retargeting is False
    flag_source_video_eye_retargeting: bool = False  # when the input is a source video, whether to let the eye-open scalar of each frame to be the same as the first source frame before the animation, only take effect when flag_eye_retargeting and flag_lip_retargeting is False, may cause the inter-frame jittering
    flag_video_editing_head_rotation: bool = False  # when the input is a source video, whether to inherit the relative head rotation from the driving video
    flag_eye_retargeting: bool = False  # not recommend to be True, WIP
    flag_lip_retargeting: bool = False  # not recommend to be True, WIP
    flag_stitching: bool = True  # recommend to True if head movement is small, False if head movement is large
    flag_relative_motion: bool = True  # whether to use relative motion
    flag_pasteback: bool = True  # whether to paste-back/stitch the animated face cropping from the face-cropping space to the original image space
    flag_do_crop: bool = True  # whether to crop the source portrait or video to the face-cropping space
    driving_smooth_observation_variance: float = 3e-6  # smooth strength scalar for the animated video when the input is a source video, the larger the number, the smoother the animated video; too much smoothness would result in loss of motion accuracy

    ########## source crop arguments ##########
    scale: float = 2.3  # the ratio of face area is smaller if scale is larger
    vx_ratio: float = 0  # the ratio to move the face to left or right in cropping space
    vy_ratio: float = -0.125  # the ratio to move the face to up or down in cropping space
    flag_do_rot: bool = True  # whether to conduct the rotation when flag_do_crop is True

    ########## driving crop arguments ##########
    scale_crop_driving_video: float = 2.2  # scale factor for cropping driving video
    vx_ratio_crop_driving_video: float = 0.  # adjust y offset
    vy_ratio_crop_driving_video: float = -0.1  # adjust x offset

    ########## gradio arguments ##########
    server_port: Annotated[int, tyro.conf.arg(aliases=["-p"])] = 8890  # port for gradio server
    share: bool = False  # whether to share the server to public
    server_name: Optional[str] = "127.0.0.1"  # set the local server name, "0.0.0.0" to broadcast all
    flag_do_torch_compile: bool = False  # whether to use torch.compile to accelerate generation
