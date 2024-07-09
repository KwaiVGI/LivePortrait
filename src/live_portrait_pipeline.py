# coding: utf-8

"""
Pipeline of LivePortrait
"""

import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)
import numpy as np
import os
import os.path as osp
from rich.progress import track

from .config.argument_config import ArgumentConfig
from .config.inference_config import InferenceConfig
from .config.crop_config import CropConfig
from .utils.cropper import Cropper
from .utils.camera import get_rotation_matrix
from .utils.video import images2video, concat_frames, get_fps, add_audio_to_video, has_audio_stream
from .utils.crop import _transform_img, prepare_paste_back, paste_back
from .utils.io import load_image_rgb, load_driving_info, resize_to_limit, dump, load
from .utils.helper import mkdir, basename, dct2device, is_video, is_template, remove_suffix
from .utils.rprint import rlog as log
# from .utils.viz import viz_lmk
from .live_portrait_wrapper import LivePortraitWrapper


def make_abs_path(fn):
    return osp.join(osp.dirname(osp.realpath(__file__)), fn)


class LivePortraitPipeline(object):

    def __init__(self, inference_cfg: InferenceConfig, crop_cfg: CropConfig):
        self.live_portrait_wrapper: LivePortraitWrapper = LivePortraitWrapper(inference_cfg=inference_cfg)
        self.cropper: Cropper = Cropper(crop_cfg=crop_cfg)

    def execute(self, args: ArgumentConfig):
        # for convenience
        inf_cfg = self.live_portrait_wrapper.inference_cfg
        device =  self.live_portrait_wrapper.device
        crop_cfg = self.cropper.crop_cfg

        ######## process source portrait ########
        img_rgb = load_image_rgb(args.source_image)
        img_rgb = resize_to_limit(img_rgb, inf_cfg.source_max_dim, inf_cfg.source_division)
        log(f"Load source image from {args.source_image}")

        crop_info = self.cropper.crop_source_image(img_rgb, crop_cfg)
        if crop_info is None:
            raise Exception("No face detected in the source image!")
        source_lmk = crop_info['lmk_crop']
        img_crop, img_crop_256x256 = crop_info['img_crop'], crop_info['img_crop_256x256']

        if inf_cfg.flag_do_crop:
            I_s = self.live_portrait_wrapper.prepare_source(img_crop_256x256)
        else:
            img_crop_256x256 = cv2.resize(img_rgb, (256, 256))  # force to resize to 256x256
            I_s = self.live_portrait_wrapper.prepare_source(img_crop_256x256)
        x_s_info = self.live_portrait_wrapper.get_kp_info(I_s)
        x_c_s = x_s_info['kp']
        R_s = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
        f_s = self.live_portrait_wrapper.extract_feature_3d(I_s)
        x_s = self.live_portrait_wrapper.transform_keypoint(x_s_info)

        flag_lip_zero = inf_cfg.flag_lip_zero  # not overwrite
        if flag_lip_zero:
            # let lip-open scalar to be 0 at first
            c_d_lip_before_animation = [0.]
            combined_lip_ratio_tensor_before_animation = self.live_portrait_wrapper.calc_combined_lip_ratio(c_d_lip_before_animation, source_lmk)
            if combined_lip_ratio_tensor_before_animation[0][0] < inf_cfg.lip_zero_threshold:
                flag_lip_zero = False
            else:
                lip_delta_before_animation = self.live_portrait_wrapper.retarget_lip(x_s, combined_lip_ratio_tensor_before_animation)
        ############################################

        ######## process driving info ########
        flag_load_from_template = is_template(args.driving_info)
        driving_rgb_crop_256x256_lst = None

        if flag_load_from_template:
            # NOTE: load from template, it is fast, but the cropping video is None
            log(f"Load from template: {args.driving_info}, NOT the video, so the cropping video and audio are both None.")
            template_dct = load(args.driving_info)
            n_frames = template_dct['n_frames']

            # set output_fps
            output_fps = template_dct.get('output_fps', inf_cfg.output_fps)

        elif osp.exists(args.driving_info) and is_video(args.driving_info):
            # load from video file, AND make motion template
            log(f"Load video: {args.driving_info}")
            if osp.isdir(args.driving_info):
                output_fps = inf_cfg.output_fps
            else:
                output_fps = int(get_fps(args.driving_info))
                log(f'The FPS of {args.driving_info} is: {output_fps}')

            log(f"Load video file (mp4 mov avi etc...): {args.driving_info}")
            driving_rgb_lst = load_driving_info(args.driving_info)

            ######## make motion template ########
            log("Start making motion template...")
            if inf_cfg.flag_crop_driving_video:
                # crop video
                ret = self.cropper.crop_driving_video(driving_rgb_lst)
                log(f'Driving video is cropped, {len(ret["frame_crop_lst"])} frames are processed.')
                driving_rgb_crop_lst, driving_lmk_crop_lst = ret['frame_crop_lst'], ret['lmk_crop_lst']
                driving_rgb_crop_256x256_lst = [cv2.resize(_, (256, 256)) for _ in driving_rgb_crop_lst]
            else:
                # not crop video
                driving_lmk_crop_lst = self.cropper.calc_lmks_from_cropped_video(driving_rgb_lst)
                driving_rgb_crop_256x256_lst = [cv2.resize(_, (256, 256)) for _ in driving_rgb_lst]  # force to resize to 256x256

            c_d_eyes_lst, c_d_lip_lst = self.live_portrait_wrapper.calc_driving_ratio(driving_lmk_crop_lst)
            # save the motion template
            I_d_lst = self.live_portrait_wrapper.prepare_driving_videos(driving_rgb_crop_256x256_lst)
            template_dct = self.make_motion_template(I_d_lst, c_d_eyes_lst, c_d_lip_lst, output_fps=output_fps)

            template_path = remove_suffix(args.driving_info) + '.pkl'
            dump(template_path, template_dct)
            log(f"Save motion template to {template_path}")

            n_frames = I_d_lst.shape[0]
        else:
            raise Exception(f"{args.driving_info} not exists or unsupported driving info types!")
        #########################################

        ######## prepare for pasteback ########
        I_p_pstbk_lst = None
        if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:
            mask_ori_float = prepare_paste_back(inf_cfg.mask_crop, crop_info['M_c2o'], dsize=(img_rgb.shape[1], img_rgb.shape[0]))
            I_p_pstbk_lst = []
            log("Prepared for pasteback")
        #########################################

        I_p_lst = []
        R_d_0, x_d_0_info = None, None

        for i in track(range(n_frames), description='Animating...', total=n_frames):
            x_d_i_info = template_dct['motion'][i]
            x_d_i_info = dct2device(x_d_i_info, device)
            R_d_i = x_d_i_info['R_d']

            if i == 0:
                R_d_0 = R_d_i
                x_d_0_info = x_d_i_info

            if inf_cfg.flag_relative_motion:
                R_new = (R_d_i @ R_d_0.permute(0, 2, 1)) @ R_s
                delta_new = x_s_info['exp'] + (x_d_i_info['exp'] - x_d_0_info['exp'])
                scale_new = x_s_info['scale'] * (x_d_i_info['scale'] / x_d_0_info['scale'])
                t_new = x_s_info['t'] + (x_d_i_info['t'] - x_d_0_info['t'])
            else:
                R_new = R_d_i
                delta_new = x_d_i_info['exp']
                scale_new = x_s_info['scale']
                t_new = x_d_i_info['t']

            t_new[..., 2].fill_(0)  # zero tz
            x_d_i_new = scale_new * (x_c_s @ R_new + delta_new) + t_new

            # Algorithm 1:
            if not inf_cfg.flag_stitching and not inf_cfg.flag_eye_retargeting and not inf_cfg.flag_lip_retargeting:
                # without stitching or retargeting
                if flag_lip_zero:
                    x_d_i_new += lip_delta_before_animation.reshape(-1, x_s.shape[1], 3)
                else:
                    pass
            elif inf_cfg.flag_stitching and not inf_cfg.flag_eye_retargeting and not inf_cfg.flag_lip_retargeting:
                # with stitching and without retargeting
                if flag_lip_zero:
                    x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new) + lip_delta_before_animation.reshape(-1, x_s.shape[1], 3)
                else:
                    x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new)
            else:
                eyes_delta, lip_delta = None, None
                if inf_cfg.flag_eye_retargeting:
                    c_d_eyes_i = c_d_eyes_lst[i]
                    combined_eye_ratio_tensor = self.live_portrait_wrapper.calc_combined_eye_ratio(c_d_eyes_i, source_lmk)
                    # ∆_eyes,i = R_eyes(x_s; c_s,eyes, c_d,eyes,i)
                    eyes_delta = self.live_portrait_wrapper.retarget_eye(x_s, combined_eye_ratio_tensor)
                if inf_cfg.flag_lip_retargeting:
                    c_d_lip_i = c_d_lip_lst[i]
                    combined_lip_ratio_tensor = self.live_portrait_wrapper.calc_combined_lip_ratio(c_d_lip_i, source_lmk)
                    # ∆_lip,i = R_lip(x_s; c_s,lip, c_d,lip,i)
                    lip_delta = self.live_portrait_wrapper.retarget_lip(x_s, combined_lip_ratio_tensor)

                if inf_cfg.flag_relative_motion:  # use x_s
                    x_d_i_new = x_s + \
                        (eyes_delta.reshape(-1, x_s.shape[1], 3) if eyes_delta is not None else 0) + \
                        (lip_delta.reshape(-1, x_s.shape[1], 3) if lip_delta is not None else 0)
                else:  # use x_d,i
                    x_d_i_new = x_d_i_new + \
                        (eyes_delta.reshape(-1, x_s.shape[1], 3) if eyes_delta is not None else 0) + \
                        (lip_delta.reshape(-1, x_s.shape[1], 3) if lip_delta is not None else 0)

                if inf_cfg.flag_stitching:
                    x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new)

            out = self.live_portrait_wrapper.warp_decode(f_s, x_s, x_d_i_new)
            I_p_i = self.live_portrait_wrapper.parse_output(out['out'])[0]
            I_p_lst.append(I_p_i)

            if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:
                # TODO: pasteback is slow, considering optimize it using multi-threading or GPU
                I_p_pstbk = paste_back(I_p_i, crop_info['M_c2o'], img_rgb, mask_ori_float)
                I_p_pstbk_lst.append(I_p_pstbk)

        mkdir(args.output_dir)
        wfp_concat = None
        flag_has_audio = (not flag_load_from_template) and has_audio_stream(args.driving_info)

        ######### build final concact result #########
        # driving frame | source image | generation, or source image | generation
        frames_concatenated = concat_frames(driving_rgb_crop_256x256_lst, img_crop_256x256, I_p_lst)
        wfp_concat = osp.join(args.output_dir, f'{basename(args.source_image)}--{basename(args.driving_info)}_concat.mp4')
        images2video(frames_concatenated, wfp=wfp_concat, fps=output_fps)

        if flag_has_audio:
            # final result with concact
            wfp_concat_with_audio = osp.join(args.output_dir, f'{basename(args.source_image)}--{basename(args.driving_info)}_concat_with_audio.mp4')
            add_audio_to_video(wfp_concat, args.driving_info, wfp_concat_with_audio)
            os.replace(wfp_concat_with_audio, wfp_concat)
            log(f"Replace {wfp_concat} with {wfp_concat_with_audio}")

        # save drived result
        wfp = osp.join(args.output_dir, f'{basename(args.source_image)}--{basename(args.driving_info)}.mp4')
        if I_p_pstbk_lst is not None and len(I_p_pstbk_lst) > 0:
            images2video(I_p_pstbk_lst, wfp=wfp, fps=output_fps)
        else:
            images2video(I_p_lst, wfp=wfp, fps=output_fps)

        ######### build final result #########
        if flag_has_audio:
            wfp_with_audio = osp.join(args.output_dir, f'{basename(args.source_image)}--{basename(args.driving_info)}_with_audio.mp4')
            add_audio_to_video(wfp, args.driving_info, wfp_with_audio)
            os.replace(wfp_with_audio, wfp)
            log(f"Replace {wfp} with {wfp_with_audio}")

        log(f'Final result: {wfp}')
        log(f'Final result with concact: {wfp_concat}')

        return wfp, wfp_concat

    def make_motion_template(self, I_d_lst, c_d_eyes_lst, c_d_lip_lst, **kwargs):
        n_frames = I_d_lst.shape[0]
        template_dct = {
            'n_frames': n_frames,
            'output_fps': kwargs.get('output_fps', 25),
            'motion': [],
            'c_d_eyes_lst': [],
            'c_d_lip_lst': [],
        }

        for i in track(range(n_frames), description='Making templates...', total=n_frames):
            # collect s_d, R_d, δ_d and t_d for inference
            I_d_i = I_d_lst[i]
            x_d_i_info = self.live_portrait_wrapper.get_kp_info(I_d_i)
            R_d_i = get_rotation_matrix(x_d_i_info['pitch'], x_d_i_info['yaw'], x_d_i_info['roll'])

            item_dct = {
                'scale': x_d_i_info['scale'].cpu().numpy().astype(np.float32),
                'R_d': R_d_i.cpu().numpy().astype(np.float32),
                'exp': x_d_i_info['exp'].cpu().numpy().astype(np.float32),
                't': x_d_i_info['t'].cpu().numpy().astype(np.float32),
            }

            template_dct['motion'].append(item_dct)

            c_d_eyes = c_d_eyes_lst[i].astype(np.float32)
            template_dct['c_d_eyes_lst'].append(c_d_eyes)

            c_d_lip = c_d_lip_lst[i].astype(np.float32)
            template_dct['c_d_lip_lst'].append(c_d_lip)

        return template_dct
