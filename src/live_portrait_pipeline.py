"""
Pipeline of LivePortrait
"""

import cv2
import numpy as np
import pickle
import os.path as osp
from rich.progress import track

from .config.argument_config import ArgumentConfig
from .config.inference_config import InferenceConfig
from .config.crop_config import CropConfig
from .utils.cropper import Cropper
from .utils.camera import get_rotation_matrix
from .utils.video import images2video, concat_frames
from .utils.crop import _transform_img, prepare_paste_back, paste_back
from .utils.retargeting_utils import calc_lip_close_ratio
from .utils.io import load_image_rgb, load_driving_info, resize_to_limit
from .utils.helper import mkdir, basename, dct2cuda, is_video, is_template
from .utils.rprint import rlog as log
from .live_portrait_wrapper import LivePortraitWrapper


def make_abs_path(fn):
    return osp.join(osp.dirname(osp.realpath(__file__)), fn)


class LivePortraitPipeline(object):

    def __init__(self, inference_cfg: InferenceConfig, crop_cfg: CropConfig):
        self.live_portrait_wrapper: LivePortraitWrapper = LivePortraitWrapper(cfg=inference_cfg)
        self.cropper = Cropper(crop_cfg=crop_cfg)

    def execute_frame(self, frame, source_image_path):
        inference_cfg = self.live_portrait_wrapper.cfg  # for convenience

        # Load and preprocess source image
        img_rgb = load_image_rgb(source_image_path)
        img_rgb = resize_to_limit(img_rgb, inference_cfg.ref_max_shape, inference_cfg.ref_shape_n)
        log(f"Load source image from {source_image_path}")
        crop_info = self.cropper.crop_single_image(img_rgb)
        source_lmk = crop_info['lmk_crop']
        img_crop, img_crop_256x256 = crop_info['img_crop'], crop_info['img_crop_256x256']

        if inference_cfg.flag_do_crop:
            I_s = self.live_portrait_wrapper.prepare_source(img_crop_256x256)
        else:
            I_s = self.live_portrait_wrapper.prepare_source(img_rgb)
        
        x_s_info = self.live_portrait_wrapper.get_kp_info(I_s)
        x_c_s = x_s_info['kp']
        R_s = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
        f_s = self.live_portrait_wrapper.extract_feature_3d(I_s)
        x_s = self.live_portrait_wrapper.transform_keypoint(x_s_info)

        lip_delta_before_animation = None
        if inference_cfg.flag_lip_zero:
            c_d_lip_before_animation = [0.]
            combined_lip_ratio_tensor_before_animation = self.live_portrait_wrapper.calc_combined_lip_ratio(c_d_lip_before_animation, crop_info['lmk_crop'])
            if combined_lip_ratio_tensor_before_animation[0][0] < inference_cfg.lip_zero_threshold:
                inference_cfg.flag_lip_zero = False
            else:
                lip_delta_before_animation = self.live_portrait_wrapper.retarget_lip(x_s, combined_lip_ratio_tensor_before_animation)

        return x_s, f_s, R_s, x_s_info, lip_delta_before_animation, crop_info, img_rgb

    def generate_frame(self, x_s, f_s, R_s, x_s_info, lip_delta_before_animation, crop_info, img_rgb, driving_info):
        inference_cfg = self.live_portrait_wrapper.cfg  # for convenience

        # Process driving info
        driving_rgb = cv2.resize(driving_info, (256, 256))
        I_d_i = self.live_portrait_wrapper.prepare_driving_videos([driving_rgb])[0]


        x_d_i_info = self.live_portrait_wrapper.get_kp_info(I_d_i)
        R_d_i = get_rotation_matrix(x_d_i_info['pitch'], x_d_i_info['yaw'], x_d_i_info['roll'])

        R_new = R_d_i @ R_s
        delta_new = x_s_info['exp'] + (x_d_i_info['exp'] - x_s_info['exp'])
        scale_new = x_s_info['scale'] * (x_d_i_info['scale'] / x_s_info['scale'])
        t_new = x_s_info['t'] + (x_d_i_info['t'] - x_s_info['t'])
        t_new[..., 2].fill_(0)  # zero tz

        x_d_i_new = scale_new * (x_s @ R_new + delta_new) + t_new
        if inference_cfg.flag_lip_zero and lip_delta_before_animation is not None:
            x_d_i_new += lip_delta_before_animation.reshape(-1, x_s.shape[1], 3)

        out = self.live_portrait_wrapper.warp_decode(f_s, x_s, x_d_i_new)
        I_p_i = self.live_portrait_wrapper.parse_output(out['out'])[0]

        if inference_cfg.flag_pasteback:
            mask_ori = prepare_paste_back(inference_cfg.mask_crop, crop_info['M_c2o'], dsize=(img_rgb.shape[1], img_rgb.shape[0]))
            I_p_i_to_ori_blend = paste_back(I_p_i, crop_info['M_c2o'], img_rgb, mask_ori)
            return I_p_i_to_ori_blend
        else:
            return I_p_i

