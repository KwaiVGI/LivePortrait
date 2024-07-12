# coding: utf-8

"""
parameters used for crop faces
"""

from dataclasses import dataclass

from .base_config import PrintableConfig


@dataclass(repr=False)  # use repr from PrintableConfig
class CropConfig(PrintableConfig):
    insightface_root: str = "../../pretrained_weights/insightface"
    landmark_ckpt_path: str = "../../pretrained_weights/liveportrait/landmark.onnx"
    device_id: int = 0  # gpu device id
    flag_force_cpu: bool = False  # force cpu inference, WIP
    ########## source image cropping option ##########
    dsize: int = 512  # crop size
    scale: float = 2.5  # scale factor
    vx_ratio: float = 0  # vx ratio
    vy_ratio: float = -0.125  # vy ratio +up, -down
    max_face_num: int = 0  # max face number, 0 mean no limit

    ########## driving video auto cropping option ##########
    scale_crop_video: float = 2.2  # 2.0 # scale factor for cropping video
    vx_ratio_crop_video: float = 0.0  # adjust y offset
    vy_ratio_crop_video: float = -0.1  # adjust x offset
    direction: str = "large-small"  # direction of cropping
