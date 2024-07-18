# coding: utf-8

"""
Pipeline of LivePortrait
"""

import torch
torch.backends.cudnn.benchmark = True # disable CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR warning

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
from .utils.io import load_image_rgb, load_video_info, resize_to_limit, dump, load
from .utils.helper import mkdir, basename, dct2device, is_video, is_template, remove_suffix, smooth
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

        ######## load source input ########
        if args.source_info.lower().endswith(('.jpg', '.jpeg', '.png')): # source input is an image
            is_source_video = False
            img_rgb = load_image_rgb(args.source_info)
            img_rgb = resize_to_limit(img_rgb, inf_cfg.source_max_dim, inf_cfg.source_division)
            log(f"Load source image from {args.source_info}")
            source_rgb_lst = [img_rgb]
        elif is_video(args.source_info): # source input is a video
            is_source_video = True
            inf_cfg.flag_relative_motion = True # set animation to be relative motion if the source input is a video
            inf_cfg.flag_eye_retargeting, inf_cfg.flag_lip_retargeting = False, False # turn off eyes and lip retargeting if the source input is a video
            source_rgb_lst = load_video_info(args.source_info)
            source_rgb_lst = [resize_to_limit(img) for img in source_rgb_lst]
            log(f"Load source video from {args.source_info}")
            source_n_frames = len(source_rgb_lst) # number of source frames
        else: # source input is an unknown format
            raise Exception(f"Unknown source format: {args.source_info}")

        ############################################

        ######## process driving info ########
        flag_load_from_template = is_template(args.driving_info)
        driving_rgb_crop_256x256_lst = None
        wfp_template = None

        if flag_load_from_template:
            # NOTE: load from template, it is fast, but the cropping video is None
            log(f"Load from template: {args.driving_info}, NOT the video, so the cropping video and audio are both NULL.", style='bold green')
            driving_template_dct = load(args.driving_info)
            driving_n_frames = driving_template_dct['n_frames'] # number of driving frames
            if is_source_video:
                n_frames = min(source_n_frames, driving_n_frames) # minimum number as the number of the animated frames

            # set output_fps
            output_fps = driving_template_dct.get('output_fps', inf_cfg.output_fps)
            log(f'The FPS of template: {output_fps}')

            if args.flag_crop_driving_video:
                log("Warning: flag_crop_driving_video is True, but the driving info is a template, so it is ignored.")

        elif osp.exists(args.driving_info) and is_video(args.driving_info):
            # load from video file, AND make motion template
            log(f"Load driving video from: {args.driving_info}")
            if osp.isdir(args.driving_info):
                output_fps = inf_cfg.output_fps
            else:
                output_fps = int(get_fps(args.driving_info))
                log(f'The FPS of {args.driving_info} is: {output_fps}')

            # log(f"Load video file (mp4 mov avi etc...): {args.driving_info}")
            driving_rgb_lst = load_video_info(args.driving_info)
            driving_n_frames = len(driving_rgb_lst) # number of driving frames

            ######## make motion template ########
            log("Start making driving motion template...")
            if is_source_video:
                n_frames = min(source_n_frames, driving_n_frames) # minimum number as the number of the animated frames
                driving_rgb_lst = driving_rgb_lst[:n_frames]
            if inf_cfg.flag_crop_driving_video:
                ret_d = self.cropper.crop_driving_video(driving_rgb_lst)
                log(f'Driving video is cropped, {len(ret_d["frame_crop_lst"])} frames are processed.')
                if len(ret_d["frame_crop_lst"]) != n_frames:
                    n_frames = min(n_frames, len(ret_d["frame_crop_lst"]))
                driving_rgb_crop_lst, driving_lmk_crop_lst = ret_d['frame_crop_lst'], ret_d['lmk_crop_lst']
                driving_rgb_crop_256x256_lst = [cv2.resize(_, (256, 256)) for _ in driving_rgb_crop_lst]
            else:
                driving_lmk_crop_lst = self.cropper.calc_lmks_from_cropped_video(driving_rgb_lst)
                driving_rgb_crop_256x256_lst = [cv2.resize(_, (256, 256)) for _ in driving_rgb_lst]  # force to resize to 256x256
            #######################################

            c_d_eyes_lst, c_d_lip_lst = self.live_portrait_wrapper.calc_ratio(driving_lmk_crop_lst)
            # save the motion template
            I_d_lst = self.live_portrait_wrapper.prepare_videos(driving_rgb_crop_256x256_lst)
            driving_template_dct = self.make_motion_template(I_d_lst, c_d_eyes_lst, c_d_lip_lst, output_fps=output_fps)

            wfp_template = remove_suffix(args.driving_info) + '.pkl'
            dump(wfp_template, driving_template_dct)
            log(f"Dump motion template to {wfp_template}")

        else:
            raise Exception(f"{args.driving_info} not exists or unsupported driving info types!")
        #########################################

        ######## prepare for pasteback ########
        I_p_pstbk_lst = None
        if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:
            I_p_pstbk_lst = []
            log("Prepared pasteback mask done.")
        #########################################

        I_p_lst = []
        R_d_0, x_d_0_info = None, None
        flag_lip_zero = inf_cfg.flag_lip_zero  # not overwrite
        flag_source_video_eye_retargeting = inf_cfg.flag_source_video_eye_retargeting # not overwrite
        lip_delta_before_animation, eye_delta_before_animation = None, None

        ######## process source info ########
        if is_source_video: # source video
            log(f"Start making source motion template...")

            source_rgb_lst = source_rgb_lst[:n_frames]
            if inf_cfg.flag_do_crop:
                ret_s = self.cropper.crop_source_video(source_rgb_lst, crop_cfg)
                log(f'Source video is cropped, {len(ret_s["frame_crop_lst"])} frames are processed.')
                if len(ret_s["frame_crop_lst"])!= n_frames:
                    n_frames = min(n_frames, len(ret_s["frame_crop_lst"]))
                img_crop_256x256_lst, source_lmk_crop_lst, source_M_c2o_lst = ret_s['frame_crop_lst'], ret_s['lmk_crop_lst'], ret_s['M_c2o_lst']
            else:
                source_lmk_crop_lst = self.cropper.calc_lmks_from_cropped_video(source_rgb_lst)
                img_crop_256x256_lst = [cv2.resize(_, (256, 256)) for _ in source_rgb_lst]  # force to resize to 256x256

            c_s_eyes_lst, c_s_lip_lst = self.live_portrait_wrapper.calc_ratio(source_lmk_crop_lst)
            # save the motion template
            I_s_lst = self.live_portrait_wrapper.prepare_videos(img_crop_256x256_lst)
            source_template_dct = self.make_motion_template(I_s_lst, c_s_eyes_lst, c_s_lip_lst, output_fps=output_fps)

            x_d_exp_lst = [source_template_dct['motion'][i]['exp'] + driving_template_dct['motion'][i]['exp'] - driving_template_dct['motion'][0]['exp'] for i in range(n_frames)]
            x_d_exp_lst_smooth = smooth(x_d_exp_lst, source_template_dct['motion'][0]['exp'].shape, device, inf_cfg.driving_smooth_observation_variance)
            if inf_cfg.flag_video_editing_head_rotation:
                key_r = 'R' if 'R' in driving_template_dct['motion'][0].keys() else 'R_d' # compatible with previous keys
                x_d_r_lst = [(np.dot(driving_template_dct['motion'][i][key_r], driving_template_dct['motion'][0][key_r].transpose(0, 2, 1))) @  source_template_dct['motion'][i][key_r] for i in range(n_frames)]
                x_d_r_lst_smooth = smooth(x_d_r_lst, source_template_dct['motion'][0][key_r].shape, device, inf_cfg.driving_smooth_observation_variance)
        else: # source image
            n_frames = driving_n_frames
            # if the input is a source image, process it only once
            crop_info = self.cropper.crop_source_image(source_rgb_lst[0], crop_cfg)
            if crop_info is None:
                raise Exception("No face detected in the source image!")
            source_lmk = crop_info['lmk_crop']
            img_crop_256x256 = crop_info['img_crop_256x256']

            if inf_cfg.flag_do_crop:
                I_s = self.live_portrait_wrapper.prepare_source(img_crop_256x256)
            else:
                img_crop_256x256 = cv2.resize(source_rgb_lst[0], (256, 256))  # force to resize to 256x256
                I_s = self.live_portrait_wrapper.prepare_source(img_crop_256x256)
            x_s_info = self.live_portrait_wrapper.get_kp_info(I_s)
            x_c_s = x_s_info['kp']
            R_s = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
            f_s = self.live_portrait_wrapper.extract_feature_3d(I_s)
            x_s = self.live_portrait_wrapper.transform_keypoint(x_s_info)

            # let lip-open scalar to be 0 at first
            if flag_lip_zero:
                c_d_lip_before_animation = [0.]
                combined_lip_ratio_tensor_before_animation = self.live_portrait_wrapper.calc_combined_lip_ratio(c_d_lip_before_animation, source_lmk)
                if combined_lip_ratio_tensor_before_animation[0][0] >= inf_cfg.lip_zero_threshold:
                    lip_delta_before_animation = self.live_portrait_wrapper.retarget_lip(x_s, combined_lip_ratio_tensor_before_animation)

            if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:
                mask_ori_float = prepare_paste_back(inf_cfg.mask_crop, crop_info['M_c2o'], dsize=(source_rgb_lst[0].shape[1], source_rgb_lst[0].shape[0]))
        #####################################

        log(f"The animated video consists of {n_frames} frames.")
        for i in track(range(n_frames), description='🚀Animating...', total=n_frames):
            if is_source_video: # source video
                x_s_info_tiny = source_template_dct['motion'][i]
                x_s_info_tiny = dct2device(x_s_info_tiny, device)

                source_lmk = source_lmk_crop_lst[i]
                img_crop_256x256 = img_crop_256x256_lst[i]
                M_c2o = source_M_c2o_lst[i]
                I_s = I_s_lst[i]

                x_s_info = source_template_dct['x_i_info_lst'][i]
                x_c_s = x_s_info['kp']
                R_s = x_s_info_tiny['R']
                f_s = self.live_portrait_wrapper.extract_feature_3d(I_s)
                x_s = self.live_portrait_wrapper.transform_keypoint(x_s_info)

                # let lip-open scalar to be 0 at first if the input is a video
                if flag_lip_zero:
                    c_d_lip_before_animation = [0.]
                    combined_lip_ratio_tensor_before_animation = self.live_portrait_wrapper.calc_combined_lip_ratio(c_d_lip_before_animation, source_lmk)
                    if combined_lip_ratio_tensor_before_animation[0][0] >= inf_cfg.lip_zero_threshold:
                        lip_delta_before_animation = self.live_portrait_wrapper.retarget_lip(x_s, combined_lip_ratio_tensor_before_animation)

                # let eye-open scalar to be the same as the first frame if the latter is eye-open state
                if flag_source_video_eye_retargeting:
                    if i == 0:
                        combined_eye_ratio_tensor_frame_zero = c_s_eyes_lst[0]
                        c_d_eye_before_animation_frame_zero = [[combined_eye_ratio_tensor_frame_zero[0][:2].mean()]]
                        if c_d_eye_before_animation_frame_zero[0][0] < inf_cfg.source_video_eye_retargeting_threshold:
                            c_d_eye_before_animation_frame_zero = [[0.39]]
                    combined_eye_ratio_tensor_before_animation = self.live_portrait_wrapper.calc_combined_eye_ratio(c_d_eye_before_animation_frame_zero, source_lmk)
                    eye_delta_before_animation = self.live_portrait_wrapper.retarget_eye(x_s, combined_eye_ratio_tensor_before_animation)

                if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching: # prepare for paste back
                    mask_ori_float = prepare_paste_back(inf_cfg.mask_crop, M_c2o, dsize=(source_rgb_lst[i].shape[1], source_rgb_lst[i].shape[0]))
                ########################################

            x_d_i_info = driving_template_dct['motion'][i]
            x_d_i_info = dct2device(x_d_i_info, device)
            R_d_i = x_d_i_info['R'] if 'R' in x_d_i_info.keys() else x_d_i_info['R_d'] # compatible with previous keys

            if i == 0:
                R_d_0 = R_d_i
                x_d_0_info = x_d_i_info

            if inf_cfg.flag_relative_motion:
                if is_source_video:
                    if inf_cfg.flag_video_editing_head_rotation:
                        R_new = x_d_r_lst_smooth[i]
                    else:
                        R_new = R_s
                else:
                    R_new = (R_d_i @ R_d_0.permute(0, 2, 1)) @ R_s

                delta_new = x_d_exp_lst_smooth[i] if is_source_video else x_s_info['exp'] + (x_d_i_info['exp'] - x_d_0_info['exp'])
                scale_new = x_s_info['scale'] if is_source_video else x_s_info['scale'] * (x_d_i_info['scale'] / x_d_0_info['scale'])
                t_new = x_s_info['t'] if is_source_video else x_s_info['t'] + (x_d_i_info['t'] - x_d_0_info['t'])
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
                if flag_lip_zero and lip_delta_before_animation != None:
                    x_d_i_new += lip_delta_before_animation
                if flag_source_video_eye_retargeting and eye_delta_before_animation != None:
                    x_d_i_new += eye_delta_before_animation
                else:
                    pass
            elif inf_cfg.flag_stitching and not inf_cfg.flag_eye_retargeting and not inf_cfg.flag_lip_retargeting:
                # with stitching and without retargeting
                if flag_lip_zero and lip_delta_before_animation != None:
                    x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new) + lip_delta_before_animation
                else:
                    x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new)
                if flag_source_video_eye_retargeting and eye_delta_before_animation != None:
                    x_d_i_new += eye_delta_before_animation
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
                        (eyes_delta if eyes_delta is not None else 0) + \
                        (lip_delta if lip_delta is not None else 0)
                else:  # use x_d,i
                    x_d_i_new = x_d_i_new + \
                        (eyes_delta if eyes_delta is not None else 0) + \
                        (lip_delta if lip_delta is not None else 0)

                if inf_cfg.flag_stitching:
                    x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new)

            out = self.live_portrait_wrapper.warp_decode(f_s, x_s, x_d_i_new)
            I_p_i = self.live_portrait_wrapper.parse_output(out['out'])[0]
            I_p_lst.append(I_p_i)

            if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:
                # TODO: the paste back procedure is slow, considering optimize it using multi-threading or GPU
                if is_source_video:
                    I_p_pstbk = paste_back(I_p_i, source_M_c2o_lst[i], source_rgb_lst[i], mask_ori_float)
                else:
                    I_p_pstbk = paste_back(I_p_i, crop_info['M_c2o'], source_rgb_lst[0], mask_ori_float)
                I_p_pstbk_lst.append(I_p_pstbk)

        mkdir(args.output_dir)
        wfp_concat = None
        flag_has_audio = (not flag_load_from_template) and has_audio_stream(args.driving_info)

        ######### build the final concatenation result #########
        # driving frame | source frame | generation, or source frame | generation
        if is_source_video:
            frames_concatenated = concat_frames(driving_rgb_crop_256x256_lst, img_crop_256x256_lst, I_p_lst)
        else:
            frames_concatenated = concat_frames(driving_rgb_crop_256x256_lst, [img_crop_256x256], I_p_lst)
        wfp_concat = osp.join(args.output_dir, f'{basename(args.source_info)}--{basename(args.driving_info)}_concat.mp4')
        images2video(frames_concatenated, wfp=wfp_concat, fps=output_fps)

        if flag_has_audio:
            # final result with concatenation
            wfp_concat_with_audio = osp.join(args.output_dir, f'{basename(args.source_info)}--{basename(args.driving_info)}_concat_with_audio.mp4')
            add_audio_to_video(wfp_concat, args.driving_info, wfp_concat_with_audio)
            os.replace(wfp_concat_with_audio, wfp_concat)
            log(f"Replace {wfp_concat} with {wfp_concat_with_audio}")

        # save the animated result
        wfp = osp.join(args.output_dir, f'{basename(args.source_info)}--{basename(args.driving_info)}.mp4')
        if I_p_pstbk_lst is not None and len(I_p_pstbk_lst) > 0:
            images2video(I_p_pstbk_lst, wfp=wfp, fps=output_fps)
        else:
            images2video(I_p_lst, wfp=wfp, fps=output_fps)

        ######### build the final result #########
        if flag_has_audio:
            wfp_with_audio = osp.join(args.output_dir, f'{basename(args.source_info)}--{basename(args.driving_info)}_with_audio.mp4')
            add_audio_to_video(wfp, args.driving_info, wfp_with_audio)
            os.replace(wfp_with_audio, wfp)
            log(f"Replace {wfp} with {wfp_with_audio}")

        # final log
        if wfp_template not in (None, ''):
            log(f'Animated template: {wfp_template}, you can specify `-d` argument with this template path next time to avoid cropping video, motion making and protecting privacy.', style='bold green')
        log(f'Animated video: {wfp}')
        log(f'Animated video with concat: {wfp_concat}')

        return wfp, wfp_concat

    def make_motion_template(self, I_lst, c_eyes_lst, c_lip_lst, **kwargs):
        n_frames = I_lst.shape[0]
        template_dct = {
            'n_frames': n_frames,
            'output_fps': kwargs.get('output_fps', 25),
            'motion': [],
            'c_eyes_lst': [],
            'c_lip_lst': [],
            'x_i_info_lst': [],
        }

        for i in track(range(n_frames), description='Making motion templates...', total=n_frames):
            # collect s, R, δ and t for inference
            I_i = I_lst[i]
            x_i_info = self.live_portrait_wrapper.get_kp_info(I_i)
            R_i = get_rotation_matrix(x_i_info['pitch'], x_i_info['yaw'], x_i_info['roll'])

            item_dct = {
                'scale': x_i_info['scale'].cpu().numpy().astype(np.float32),
                'R': R_i.cpu().numpy().astype(np.float32),
                'exp': x_i_info['exp'].cpu().numpy().astype(np.float32),
                't': x_i_info['t'].cpu().numpy().astype(np.float32),
            }

            template_dct['motion'].append(item_dct)

            c_eyes = c_eyes_lst[i].astype(np.float32)
            template_dct['c_eyes_lst'].append(c_eyes)

            c_lip = c_lip_lst[i].astype(np.float32)
            template_dct['c_lip_lst'].append(c_lip)

            template_dct['x_i_info_lst'].append(x_i_info)

        return template_dct
