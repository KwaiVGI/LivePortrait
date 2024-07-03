# coding: utf-8

"""
Pipeline of LivePortrait
"""

# TODO:
# 1. 当前假定所有的模板都是已经裁好的，需要修改下
# 2. pick样例图 source + driving

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
from .utils.crop import _transform_img
from .utils.retargeting_utils import calc_lip_close_ratio
from .utils.io import load_image_rgb, load_driving_info
from .utils.helper import mkdir, basename, dct2cuda, is_video, is_template, resize_to_limit
from .utils.rprint import rlog as log
from .live_portrait_wrapper import LivePortraitWrapper


def make_abs_path(fn):
    return osp.join(osp.dirname(osp.realpath(__file__)), fn)


class LivePortraitPipeline(object):

    def __init__(self, inference_cfg: InferenceConfig, crop_cfg: CropConfig):
        self.live_portrait_wrapper: LivePortraitWrapper = LivePortraitWrapper(cfg=inference_cfg)
        self.cropper = Cropper(crop_cfg=crop_cfg)

    def execute(self, args: ArgumentConfig):
        inference_cfg = self.live_portrait_wrapper.cfg # for convenience
        ######## process reference portrait ########
        img_rgb = load_image_rgb(args.source_image)
        img_rgb = resize_to_limit(img_rgb, inference_cfg.ref_max_shape, inference_cfg.ref_shape_n)
        log(f"Load source image from {args.source_image}")
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

        if inference_cfg.flag_lip_zero:
            # let lip-open scalar to be 0 at first
            c_d_lip_before_animation = [0.]
            combined_lip_ratio_tensor_before_animation = self.live_portrait_wrapper.calc_combined_lip_ratio(c_d_lip_before_animation, source_lmk)
            if combined_lip_ratio_tensor_before_animation[0][0] < inference_cfg.lip_zero_threshold:
                inference_cfg.flag_lip_zero = False
            else:
                lip_delta_before_animation = self.live_portrait_wrapper.retarget_lip(x_s, combined_lip_ratio_tensor_before_animation)
        ############################################

        ######## process driving info ########
        if is_video(args.driving_info):
            log(f"Load from video file (mp4 mov avi etc...): {args.driving_info}")
            # TODO: 这里track一下驱动视频 -> 构建模板
            driving_rgb_lst = load_driving_info(args.driving_info)
            driving_rgb_lst_256 = [cv2.resize(_, (256, 256)) for _ in driving_rgb_lst]
            I_d_lst = self.live_portrait_wrapper.prepare_driving_videos(driving_rgb_lst_256)
            n_frames = I_d_lst.shape[0]
            if inference_cfg.flag_eye_retargeting or inference_cfg.flag_lip_retargeting:
                driving_lmk_lst = self.cropper.get_retargeting_lmk_info(driving_rgb_lst)
                input_eye_ratio_lst, input_lip_ratio_lst = self.live_portrait_wrapper.calc_retargeting_ratio(source_lmk, driving_lmk_lst)
        elif is_template(args.driving_info):
            log(f"Load from video templates {args.driving_info}")
            with open(args.driving_info, 'rb') as f:
                template_lst, driving_lmk_lst = pickle.load(f)
            n_frames = template_lst[0]['n_frames']
            input_eye_ratio_lst, input_lip_ratio_lst = self.live_portrait_wrapper.calc_retargeting_ratio(source_lmk, driving_lmk_lst)
        else:
            raise Exception("Unsupported driving types!")
        #########################################

        ######## prepare for pasteback ########
        if inference_cfg.flag_pasteback:
            if inference_cfg.mask_crop is None:
                inference_cfg.mask_crop = cv2.imread(make_abs_path('./utils/resources/mask_template.png'), cv2.IMREAD_COLOR)
            mask_ori = _transform_img(inference_cfg.mask_crop, crop_info['M_c2o'], dsize=(img_rgb.shape[1], img_rgb.shape[0]))
            mask_ori = mask_ori.astype(np.float32) / 255.
            I_p_paste_lst = []
        #########################################

        I_p_lst = []
        R_d_0, x_d_0_info = None, None
        for i in track(range(n_frames), description='Animating...', total=n_frames):
            if is_video(args.driving_info):
                # extract kp info by M
                I_d_i = I_d_lst[i]
                x_d_i_info = self.live_portrait_wrapper.get_kp_info(I_d_i)
                R_d_i = get_rotation_matrix(x_d_i_info['pitch'], x_d_i_info['yaw'], x_d_i_info['roll'])
            else:
                # from template
                x_d_i_info = template_lst[i]
                x_d_i_info = dct2cuda(x_d_i_info, inference_cfg.device_id)
                R_d_i = x_d_i_info['R_d']

            if i == 0:
                R_d_0 = R_d_i
                x_d_0_info = x_d_i_info

            if inference_cfg.flag_relative:
                R_new = (R_d_i @ R_d_0.permute(0, 2, 1)) @ R_s
                delta_new = x_s_info['exp'] + (x_d_i_info['exp'] - x_d_0_info['exp'])
                scale_new = x_s_info['scale'] * (x_d_i_info['scale'] / x_d_0_info['scale'])
                t_new = x_s_info['t'] + (x_d_i_info['t'] - x_d_0_info['t'])
            else:
                R_new = R_d_i
                delta_new = x_d_i_info['exp']
                scale_new = x_s_info['scale']
                t_new = x_d_i_info['t']

            t_new[..., 2].fill_(0) # zero tz
            x_d_i_new = scale_new * (x_c_s @ R_new + delta_new) + t_new

            # Algorithm 1:
            if not inference_cfg.flag_stitching and not inference_cfg.flag_eye_retargeting and not inference_cfg.flag_lip_retargeting:
                # without stitching or retargeting
                if inference_cfg.flag_lip_zero:
                    x_d_i_new += lip_delta_before_animation.reshape(-1, x_s.shape[1], 3)
                else:
                    pass
            elif inference_cfg.flag_stitching and not inference_cfg.flag_eye_retargeting and not inference_cfg.flag_lip_retargeting:
                # with stitching and without retargeting
                if inference_cfg.flag_lip_zero:
                    x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new) + lip_delta_before_animation.reshape(-1, x_s.shape[1], 3)
                else:
                    x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new)
            else:
                eyes_delta, lip_delta = None, None
                if inference_cfg.flag_eye_retargeting:
                    c_d_eyes_i = input_eye_ratio_lst[i]
                    combined_eye_ratio_tensor = self.live_portrait_wrapper.calc_combined_eye_ratio(c_d_eyes_i, source_lmk)
                    # ∆_eyes,i = R_eyes(x_s; c_s,eyes, c_d,eyes,i)
                    eyes_delta = self.live_portrait_wrapper.retarget_eye(x_s, combined_eye_ratio_tensor)
                if inference_cfg.flag_lip_retargeting:
                    c_d_lip_i = input_lip_ratio_lst[i]
                    combined_lip_ratio_tensor = self.live_portrait_wrapper.calc_combined_lip_ratio(c_d_lip_i, source_lmk)
                    # ∆_lip,i = R_lip(x_s; c_s,lip, c_d,lip,i)
                    lip_delta = self.live_portrait_wrapper.retarget_lip(x_s, combined_lip_ratio_tensor)

                if inference_cfg.flag_relative:  # use x_s
                    x_d_i_new = x_s + \
                        (eyes_delta.reshape(-1, x_s.shape[1], 3) if eyes_delta is not None else 0) + \
                        (lip_delta.reshape(-1, x_s.shape[1], 3) if lip_delta is not None else 0)
                else:  # use x_d,i
                    x_d_i_new = x_d_i_new + \
                        (eyes_delta.reshape(-1, x_s.shape[1], 3) if eyes_delta is not None else 0) + \
                        (lip_delta.reshape(-1, x_s.shape[1], 3) if lip_delta is not None else 0)

                if inference_cfg.flag_stitching:
                    x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new)

            out = self.live_portrait_wrapper.warp_decode(f_s, x_s, x_d_i_new)
            I_p_i = self.live_portrait_wrapper.parse_output(out['out'])[0]
            I_p_lst.append(I_p_i)

            if inference_cfg.flag_pasteback:
                I_p_i_to_ori = _transform_img(I_p_i, crop_info['M_c2o'], dsize=(img_rgb.shape[1], img_rgb.shape[0]))
                I_p_i_to_ori_blend = np.clip(mask_ori * I_p_i_to_ori + (1 - mask_ori) * img_rgb, 0, 255).astype(np.uint8)
                out = np.hstack([I_p_i_to_ori, I_p_i_to_ori_blend])
                I_p_paste_lst.append(I_p_i_to_ori_blend)

        mkdir(args.output_dir)
        wfp_concat = None
        if is_video(args.driving_info):
            frames_concatenated = concat_frames(I_p_lst, driving_rgb_lst, img_crop_256x256)
            # save (driving frames, source image, drived frames) result
            wfp_concat = osp.join(args.output_dir, f'{basename(args.source_image)}--{basename(args.driving_info)}_concat.mp4')
            images2video(frames_concatenated, wfp=wfp_concat)

        # save drived result
        wfp = osp.join(args.output_dir, f'{basename(args.source_image)}--{basename(args.driving_info)}.mp4')
        if inference_cfg.flag_pasteback:
            images2video(I_p_paste_lst, wfp=wfp)
        else:
            images2video(I_p_lst, wfp=wfp)

        return wfp, wfp_concat
