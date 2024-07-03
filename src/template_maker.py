# coding: utf-8

"""
Make video template
"""

import os
import cv2
import numpy as np
import pickle
from rich.progress import track
from .utils.cropper import Cropper

from .utils.io import load_driving_info
from .utils.camera import get_rotation_matrix
from .utils.helper import mkdir, basename
from .utils.rprint import rlog as log
from .config.crop_config import CropConfig
from .config.inference_config import InferenceConfig
from .live_portrait_wrapper import LivePortraitWrapper

class TemplateMaker:

    def __init__(self, inference_cfg: InferenceConfig, crop_cfg: CropConfig):
        self.live_portrait_wrapper: LivePortraitWrapper = LivePortraitWrapper(cfg=inference_cfg)
        self.cropper = Cropper(crop_cfg=crop_cfg)

    def make_motion_template(self, video_fp: str, output_path: str, **kwargs):
        """ make video template (.pkl format)
        video_fp: driving video file path
        output_path: where to save the pickle file
        """

        driving_rgb_lst = load_driving_info(video_fp)
        driving_rgb_lst = [cv2.resize(_, (256, 256)) for _ in driving_rgb_lst]
        driving_lmk_lst = self.cropper.get_retargeting_lmk_info(driving_rgb_lst)
        I_d_lst = self.live_portrait_wrapper.prepare_driving_videos(driving_rgb_lst)

        n_frames = I_d_lst.shape[0]

        templates = []


        for i in track(range(n_frames), description='Making templates...', total=n_frames):
            I_d_i = I_d_lst[i]
            x_d_i_info = self.live_portrait_wrapper.get_kp_info(I_d_i)
            R_d_i = get_rotation_matrix(x_d_i_info['pitch'], x_d_i_info['yaw'], x_d_i_info['roll'])
            # collect s_d, R_d, δ_d and t_d for inference
            template_dct = {
                'n_frames': n_frames,
                'frames_index': i,
            }
            template_dct['scale'] = x_d_i_info['scale'].cpu().numpy().astype(np.float32)
            template_dct['R_d'] = R_d_i.cpu().numpy().astype(np.float32)
            template_dct['exp'] = x_d_i_info['exp'].cpu().numpy().astype(np.float32)
            template_dct['t'] = x_d_i_info['t'].cpu().numpy().astype(np.float32)

            templates.append(template_dct)

        mkdir(output_path)
        # Save the dictionary as a pickle file
        pickle_fp = os.path.join(output_path, f'{basename(video_fp)}.pkl')
        with open(pickle_fp, 'wb') as f:
            pickle.dump([templates, driving_lmk_lst], f)
        log(f"Template saved at {pickle_fp}")
