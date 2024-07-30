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
    xpose_config_file: str = "../utils/dependencies/XPose/config_model/UniPose_SwinT.py"
    xpose_ckpt_path: str = "../../pretrained_weights/liveportrait/animal_landmark.pth"
    device_id: int = 0  # gpu device id
    flag_force_cpu: bool = False  # force cpu inference, WIP
    det_thresh: float = 0.1 # detection threshold
    det_type: str = "insight" # "x" or "insight", "insight" only for human
    ########## source image or video cropping option ##########
    dsize: int = 512  # crop size
    scale: float = 2.8  # scale factor
    vx_ratio: float = 0  # vx ratio
    vy_ratio: float = -0.125  # vy ratio +up, -down
    max_face_num: int = 0  # max face number, 0 mean no limit
    flag_do_rot: bool = True # whether to conduct the rotation when flag_do_crop is True

    ########## driving video auto cropping option ##########
    scale_crop_driving_video: float = 2.2  # 2.0 # scale factor for cropping driving video
    vx_ratio_crop_driving_video: float = 0.0  # adjust y offset
    vy_ratio_crop_driving_video: float = -0.1  # adjust x offset
    direction: str = "large-small"  # direction of cropping
