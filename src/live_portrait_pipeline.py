# coding: utf-8

"""
Pipeline of LivePortrait (Human)
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
from .utils.crop import prepare_paste_back, paste_back
from .utils.io import load_image_rgb, load_video, resize_to_limit, dump, load
from .utils.helper import mkdir, basename, dct2device, is_video, is_template, remove_suffix, is_image, is_square_video, calc_motion_multiplier
from .utils.filter import smooth
from .utils.rprint import rlog as log
# from .utils.viz import viz_lmk
from .live_portrait_wrapper import LivePortraitWrapper


def make_abs_path(fn):
    return osp.join(osp.dirname(osp.realpath(__file__)), fn)


class LivePortraitPipeline(object):

    def __init__(self, inference_cfg: InferenceConfig, crop_cfg: CropConfig):
        self.live_portrait_wrapper: LivePortraitWrapper = LivePortraitWrapper(inference_cfg=inference_cfg)
        self.cropper: Cropper = Cropper(crop_cfg=crop_cfg)

    def make_motion_template(self, I_lst, c_eyes_lst, c_lip_lst, **kwargs):
        n_frames = I_lst.shape[0]
        template_dct = {
            'n_frames': n_frames,
            'output_fps': kwargs.get('output_fps', 25),
            'motion': [],
            'c_eyes_lst': [],
            'c_lip_lst': [],
        }

        for i in track(range(n_frames), description='Making motion templates...', total=n_frames):
            # collect s, R, δ and t for inference
            I_i = I_lst[i]
            x_i_info = self.live_portrait_wrapper.get_kp_info(I_i)
            x_s = self.live_portrait_wrapper.transform_keypoint(x_i_info)
            R_i = get_rotation_matrix(x_i_info['pitch'], x_i_info['yaw'], x_i_info['roll'])

            item_dct = {
                'scale': x_i_info['scale'].cpu().numpy().astype(np.float32),
                'R': R_i.cpu().numpy().astype(np.float32),
                'exp': x_i_info['exp'].cpu().numpy().astype(np.float32),
                't': x_i_info['t'].cpu().numpy().astype(np.float32),
                'kp': x_i_info['kp'].cpu().numpy().astype(np.float32),
                'x_s': x_s.cpu().numpy().astype(np.float32),
            }

            template_dct['motion'].append(item_dct)

            c_eyes = c_eyes_lst[i].astype(np.float32)
            template_dct['c_eyes_lst'].append(c_eyes)

            c_lip = c_lip_lst[i].astype(np.float32)
            template_dct['c_lip_lst'].append(c_lip)

        return template_dct

    def execute(self, args: ArgumentConfig):
        # for convenience
        inf_cfg = self.live_portrait_wrapper.inference_cfg
        device = self.live_portrait_wrapper.device
        crop_cfg = self.cropper.crop_cfg

        ######## load source input ########
        flag_is_source_video = False
        source_fps = None
        if is_image(args.source):
            flag_is_source_video = False
            img_rgb = load_image_rgb(args.source)
            img_rgb = resize_to_limit(img_rgb, inf_cfg.source_max_dim, inf_cfg.source_division)
            log(f"Load source image from {args.source}")
            source_rgb_lst = [img_rgb]
        elif is_video(args.source):
            flag_is_source_video = True
            source_rgb_lst = load_video(args.source)
            source_rgb_lst = [resize_to_limit(img, inf_cfg.source_max_dim, inf_cfg.source_division) for img in source_rgb_lst]
            source_fps = get_fps(args.source)
            log(f"Load source video from {args.source}, FPS is {source_fps}")
        else:  # source input is an unknown format
            raise Exception(f"Unknown source format: {args.source}")

        ######## process driving info ########
        flag_load_from_template = is_template(args.driving)
        driving_rgb_crop_256x256_lst = None
        wfp_template = None

        if flag_load_from_template:
            # NOTE: load from template, it is fast, but the cropping video is None
            log(f"Load from template: {args.driving}, NOT the video, so the cropping video and audio are both NULL.", style='bold green')
            driving_template_dct = load(args.driving)
            c_d_eyes_lst = driving_template_dct['c_eyes_lst'] if 'c_eyes_lst' in driving_template_dct.keys() else driving_template_dct['c_d_eyes_lst'] # compatible with previous keys
            c_d_lip_lst = driving_template_dct['c_lip_lst'] if 'c_lip_lst' in driving_template_dct.keys() else driving_template_dct['c_d_lip_lst']
            driving_n_frames = driving_template_dct['n_frames']
            flag_is_driving_video = True if driving_n_frames > 1 else False
            if flag_is_source_video and flag_is_driving_video:
                n_frames = min(len(source_rgb_lst), driving_n_frames)  # minimum number as the number of the animated frames
            elif flag_is_source_video and not flag_is_driving_video:
                n_frames = len(source_rgb_lst)
            else:
                n_frames = driving_n_frames

            # set output_fps
            output_fps = driving_template_dct.get('output_fps', inf_cfg.output_fps)
            log(f'The FPS of template: {output_fps}')

            if args.flag_crop_driving_video:
                log("Warning: flag_crop_driving_video is True, but the driving info is a template, so it is ignored.")

        elif osp.exists(args.driving):
            if is_video(args.driving):
                flag_is_driving_video = True
                # load from video file, AND make motion template
                output_fps = get_fps(args.driving)
                log(f"Load driving video from: {args.driving}, FPS is {output_fps}")
                driving_rgb_lst = load_video(args.driving)
            elif is_image(args.driving):
                flag_is_driving_video = False
                driving_img_rgb = load_image_rgb(args.driving)
                output_fps = 25
                log(f"Load driving image from {args.driving}")
                driving_rgb_lst = [driving_img_rgb]
            else:
                raise Exception(f"{args.driving} is not a supported type!")
            ######## make motion template ########
            log("Start making driving motion template...")
            driving_n_frames = len(driving_rgb_lst)
            if flag_is_source_video and flag_is_driving_video:
                n_frames = min(len(source_rgb_lst), driving_n_frames)  # minimum number as the number of the animated frames
                driving_rgb_lst = driving_rgb_lst[:n_frames]
            elif flag_is_source_video and not flag_is_driving_video:
                n_frames = len(source_rgb_lst)
            else:
                n_frames = driving_n_frames
            if inf_cfg.flag_crop_driving_video or (not is_square_video(args.driving)):
                ret_d = self.cropper.crop_driving_video(driving_rgb_lst)
                log(f'Driving video is cropped, {len(ret_d["frame_crop_lst"])} frames are processed.')
                if len(ret_d["frame_crop_lst"]) is not n_frames and flag_is_driving_video:
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

            wfp_template = remove_suffix(args.driving) + '.pkl'
            dump(wfp_template, driving_template_dct)
            log(f"Dump motion template to {wfp_template}")
        else:
            raise Exception(f"{args.driving} does not exist!")
        if not flag_is_driving_video:
            c_d_eyes_lst = c_d_eyes_lst*n_frames
            c_d_lip_lst = c_d_lip_lst*n_frames

        ######## prepare for pasteback ########
        I_p_pstbk_lst = None
        if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:
            I_p_pstbk_lst = []
            log("Prepared pasteback mask done.")

        I_p_lst = []
        R_d_0, x_d_0_info = None, None
        flag_normalize_lip = inf_cfg.flag_normalize_lip  # not overwrite
        flag_source_video_eye_retargeting = inf_cfg.flag_source_video_eye_retargeting  # not overwrite
        lip_delta_before_animation, eye_delta_before_animation = None, None

        ######## process source info ########
        if flag_is_source_video:
            log(f"Start making source motion template...")

            source_rgb_lst = source_rgb_lst[:n_frames]
            if inf_cfg.flag_do_crop:
                ret_s = self.cropper.crop_source_video(source_rgb_lst, crop_cfg)
                log(f'Source video is cropped, {len(ret_s["frame_crop_lst"])} frames are processed.')
                if len(ret_s["frame_crop_lst"]) is not n_frames:
                    n_frames = min(n_frames, len(ret_s["frame_crop_lst"]))
                img_crop_256x256_lst, source_lmk_crop_lst, source_M_c2o_lst = ret_s['frame_crop_lst'], ret_s['lmk_crop_lst'], ret_s['M_c2o_lst']
            else:
                source_lmk_crop_lst = self.cropper.calc_lmks_from_cropped_video(source_rgb_lst)
                img_crop_256x256_lst = [cv2.resize(_, (256, 256)) for _ in source_rgb_lst]  # force to resize to 256x256

            c_s_eyes_lst, c_s_lip_lst = self.live_portrait_wrapper.calc_ratio(source_lmk_crop_lst)
            # save the motion template
            I_s_lst = self.live_portrait_wrapper.prepare_videos(img_crop_256x256_lst)
            source_template_dct = self.make_motion_template(I_s_lst, c_s_eyes_lst, c_s_lip_lst, output_fps=source_fps)

            key_r = 'R' if 'R' in driving_template_dct['motion'][0].keys() else 'R_d'  # compatible with previous keys
            if inf_cfg.flag_relative_motion:
                if flag_is_driving_video:
                    x_d_exp_lst = [source_template_dct['motion'][i]['exp'] + driving_template_dct['motion'][i]['exp'] - driving_template_dct['motion'][0]['exp'] for i in range(n_frames)]
                    x_d_exp_lst_smooth = smooth(x_d_exp_lst, source_template_dct['motion'][0]['exp'].shape, device, inf_cfg.driving_smooth_observation_variance)
                else:
                    x_d_exp_lst = [source_template_dct['motion'][i]['exp'] + (driving_template_dct['motion'][0]['exp'] - inf_cfg.lip_array) for i in range(n_frames)]
                    x_d_exp_lst_smooth = [torch.tensor(x_d_exp[0], dtype=torch.float32, device=device) for x_d_exp in x_d_exp_lst]
                if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "pose":
                    if flag_is_driving_video:
                        x_d_r_lst = [(np.dot(driving_template_dct['motion'][i][key_r], driving_template_dct['motion'][0][key_r].transpose(0, 2, 1))) @ source_template_dct['motion'][i]['R'] for i in range(n_frames)]
                        x_d_r_lst_smooth = smooth(x_d_r_lst, source_template_dct['motion'][0]['R'].shape, device, inf_cfg.driving_smooth_observation_variance)
                    else:
                        x_d_r_lst = [source_template_dct['motion'][i]['R'] for i in range(n_frames)]
                        x_d_r_lst_smooth = [torch.tensor(x_d_r[0], dtype=torch.float32, device=device) for x_d_r in x_d_r_lst]
            else:
                if flag_is_driving_video:
                    x_d_exp_lst = [driving_template_dct['motion'][i]['exp'] for i in range(n_frames)]
                    x_d_exp_lst_smooth = smooth(x_d_exp_lst, source_template_dct['motion'][0]['exp'].shape, device, inf_cfg.driving_smooth_observation_variance)
                else:
                    x_d_exp_lst = [driving_template_dct['motion'][0]['exp']]
                    x_d_exp_lst_smooth = [torch.tensor(x_d_exp[0], dtype=torch.float32, device=device) for x_d_exp in x_d_exp_lst]*n_frames
                if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "pose":
                    if flag_is_driving_video:
                        x_d_r_lst = [driving_template_dct['motion'][i][key_r] for i in range(n_frames)]
                        x_d_r_lst_smooth = smooth(x_d_r_lst, source_template_dct['motion'][0]['R'].shape, device, inf_cfg.driving_smooth_observation_variance)
                    else:
                        x_d_r_lst = [driving_template_dct['motion'][0][key_r]]
                        x_d_r_lst_smooth = [torch.tensor(x_d_r[0], dtype=torch.float32, device=device) for x_d_r in x_d_r_lst]*n_frames

        else:  # if the input is a source image, process it only once
            if inf_cfg.flag_do_crop:
                crop_info = self.cropper.crop_source_image(source_rgb_lst[0], crop_cfg)
                if crop_info is None:
                    raise Exception("No face detected in the source image!")
                source_lmk = crop_info['lmk_crop']
                img_crop_256x256 = crop_info['img_crop_256x256']
            else:
                source_lmk = self.cropper.calc_lmk_from_cropped_image(source_rgb_lst[0])
                img_crop_256x256 = cv2.resize(source_rgb_lst[0], (256, 256))  # force to resize to 256x256
            I_s = self.live_portrait_wrapper.prepare_source(img_crop_256x256)
            x_s_info = self.live_portrait_wrapper.get_kp_info(I_s)
            x_c_s = x_s_info['kp']
            R_s = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
            f_s = self.live_portrait_wrapper.extract_feature_3d(I_s)
            x_s = self.live_portrait_wrapper.transform_keypoint(x_s_info)

            # let lip-open scalar to be 0 at first
            if flag_normalize_lip and inf_cfg.flag_relative_motion and source_lmk is not None:
                c_d_lip_before_animation = [0.]
                combined_lip_ratio_tensor_before_animation = self.live_portrait_wrapper.calc_combined_lip_ratio(c_d_lip_before_animation, source_lmk)
                if combined_lip_ratio_tensor_before_animation[0][0] >= inf_cfg.lip_normalize_threshold:
                    lip_delta_before_animation = self.live_portrait_wrapper.retarget_lip(x_s, combined_lip_ratio_tensor_before_animation)

            if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:
                mask_ori_float = prepare_paste_back(inf_cfg.mask_crop, crop_info['M_c2o'], dsize=(source_rgb_lst[0].shape[1], source_rgb_lst[0].shape[0]))

        ######## animate ########
        if flag_is_driving_video or (flag_is_source_video and not flag_is_driving_video):
            log(f"The animated video consists of {n_frames} frames.")
        else:
            log(f"The output of image-driven portrait animation is an image.")
        for i in track(range(n_frames), description='🚀Animating...', total=n_frames):
            if flag_is_source_video:  # source video
                x_s_info = source_template_dct['motion'][i]
                x_s_info = dct2device(x_s_info, device)

                source_lmk = source_lmk_crop_lst[i]
                img_crop_256x256 = img_crop_256x256_lst[i]
                I_s = I_s_lst[i]
                f_s = self.live_portrait_wrapper.extract_feature_3d(I_s)

                x_c_s = x_s_info['kp']
                R_s = x_s_info['R']
                x_s =x_s_info['x_s']

                # let lip-open scalar to be 0 at first if the input is a video
                if flag_normalize_lip and inf_cfg.flag_relative_motion and source_lmk is not None:
                    c_d_lip_before_animation = [0.]
                    combined_lip_ratio_tensor_before_animation = self.live_portrait_wrapper.calc_combined_lip_ratio(c_d_lip_before_animation, source_lmk)
                    if combined_lip_ratio_tensor_before_animation[0][0] >= inf_cfg.lip_normalize_threshold:
                        lip_delta_before_animation = self.live_portrait_wrapper.retarget_lip(x_s, combined_lip_ratio_tensor_before_animation)
                    else:
                        lip_delta_before_animation = None

                # let eye-open scalar to be the same as the first frame if the latter is eye-open state
                if flag_source_video_eye_retargeting and source_lmk is not None:
                    if i == 0:
                        combined_eye_ratio_tensor_frame_zero = c_s_eyes_lst[0]
                        c_d_eye_before_animation_frame_zero = [[combined_eye_ratio_tensor_frame_zero[0][:2].mean()]]
                        if c_d_eye_before_animation_frame_zero[0][0] < inf_cfg.source_video_eye_retargeting_threshold:
                            c_d_eye_before_animation_frame_zero = [[0.39]]
                    combined_eye_ratio_tensor_before_animation = self.live_portrait_wrapper.calc_combined_eye_ratio(c_d_eye_before_animation_frame_zero, source_lmk)
                    eye_delta_before_animation = self.live_portrait_wrapper.retarget_eye(x_s, combined_eye_ratio_tensor_before_animation)

                if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:  # prepare for paste back
                    mask_ori_float = prepare_paste_back(inf_cfg.mask_crop, source_M_c2o_lst[i], dsize=(source_rgb_lst[i].shape[1], source_rgb_lst[i].shape[0]))
            if flag_is_source_video and not flag_is_driving_video:
                x_d_i_info = driving_template_dct['motion'][0]
            else:
                x_d_i_info = driving_template_dct['motion'][i]
            x_d_i_info = dct2device(x_d_i_info, device)
            R_d_i = x_d_i_info['R'] if 'R' in x_d_i_info.keys() else x_d_i_info['R_d']  # compatible with previous keys

            if i == 0:  # cache the first frame
                R_d_0 = R_d_i
                x_d_0_info = x_d_i_info.copy()

            delta_new = x_s_info['exp'].clone()
            if inf_cfg.flag_relative_motion:
                if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "pose":
                    R_new = x_d_r_lst_smooth[i] if flag_is_source_video else (R_d_i @ R_d_0.permute(0, 2, 1)) @ R_s
                else:
                    R_new = R_s
                if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "exp":
                    if flag_is_source_video:
                        for idx in [1,2,6,11,12,13,14,15,16,17,18,19,20]:
                            delta_new[:, idx, :] = x_d_exp_lst_smooth[i][idx, :]
                        delta_new[:, 3:5, 1] = x_d_exp_lst_smooth[i][3:5, 1]
                        delta_new[:, 5, 2] = x_d_exp_lst_smooth[i][5, 2]
                        delta_new[:, 8, 2] = x_d_exp_lst_smooth[i][8, 2]
                        delta_new[:, 9, 1:] = x_d_exp_lst_smooth[i][9, 1:]
                    else:
                        if flag_is_driving_video:
                            delta_new = x_s_info['exp'] + (x_d_i_info['exp'] - x_d_0_info['exp'])
                        else:
                            delta_new = x_s_info['exp'] + (x_d_i_info['exp'] - torch.from_numpy(inf_cfg.lip_array).to(dtype=torch.float32, device=device))
                elif inf_cfg.animation_region == "lip":
                    for lip_idx in [6, 12, 14, 17, 19, 20]:
                        if flag_is_source_video:
                            delta_new[:, lip_idx, :] = x_d_exp_lst_smooth[i][lip_idx, :]
                        elif flag_is_driving_video:
                            delta_new[:, lip_idx, :] = (x_s_info['exp'] + (x_d_i_info['exp'] - x_d_0_info['exp']))[:, lip_idx, :]
                        else:
                            delta_new[:, lip_idx, :] = (x_s_info['exp'] + (x_d_i_info['exp'] - torch.from_numpy(inf_cfg.lip_array).to(dtype=torch.float32, device=device)))[:, lip_idx, :]
                elif inf_cfg.animation_region == "eyes":
                    for eyes_idx in [11, 13, 15, 16, 18]:
                        if flag_is_source_video:
                            delta_new[:, eyes_idx, :] = x_d_exp_lst_smooth[i][eyes_idx, :]
                        elif flag_is_driving_video:
                            delta_new[:, eyes_idx, :] = (x_s_info['exp'] + (x_d_i_info['exp'] - x_d_0_info['exp']))[:, eyes_idx, :]
                        else:
                            delta_new[:, eyes_idx, :] = (x_s_info['exp'] + (x_d_i_info['exp'] - 0))[:, eyes_idx, :]
                if inf_cfg.animation_region == "all":
                    scale_new = x_s_info['scale'] if flag_is_source_video else x_s_info['scale'] * (x_d_i_info['scale'] / x_d_0_info['scale'])
                else:
                    scale_new = x_s_info['scale']
                if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "pose":
                    t_new = x_s_info['t'] if flag_is_source_video else x_s_info['t'] + (x_d_i_info['t'] - x_d_0_info['t'])
                else:
                    t_new = x_s_info['t']
            else:
                if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "pose":
                    R_new = x_d_r_lst_smooth[i] if flag_is_source_video else R_d_i
                else:
                    R_new = R_s
                if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "exp":
                    for idx in [1,2,6,11,12,13,14,15,16,17,18,19,20]:
                        delta_new[:, idx, :] = x_d_exp_lst_smooth[i][idx, :] if flag_is_source_video else x_d_i_info['exp'][:, idx, :]
                    delta_new[:, 3:5, 1] = x_d_exp_lst_smooth[i][3:5, 1] if flag_is_source_video else x_d_i_info['exp'][:, 3:5, 1]
                    delta_new[:, 5, 2] = x_d_exp_lst_smooth[i][5, 2] if flag_is_source_video else x_d_i_info['exp'][:, 5, 2]
                    delta_new[:, 8, 2] = x_d_exp_lst_smooth[i][8, 2] if flag_is_source_video else x_d_i_info['exp'][:, 8, 2]
                    delta_new[:, 9, 1:] = x_d_exp_lst_smooth[i][9, 1:] if flag_is_source_video else x_d_i_info['exp'][:, 9, 1:]
                elif inf_cfg.animation_region == "lip":
                    for lip_idx in [6, 12, 14, 17, 19, 20]:
                        delta_new[:, lip_idx, :] = x_d_exp_lst_smooth[i][lip_idx, :] if flag_is_source_video else x_d_i_info['exp'][:, lip_idx, :]
                elif inf_cfg.animation_region == "eyes":
                    for eyes_idx in [11, 13, 15, 16, 18]:
                        delta_new[:, eyes_idx, :] = x_d_exp_lst_smooth[i][eyes_idx, :] if flag_is_source_video else x_d_i_info['exp'][:, eyes_idx, :]
                scale_new = x_s_info['scale']
                if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "pose":
                    t_new = x_d_i_info['t']
                else:
                    t_new = x_s_info['t']

            t_new[..., 2].fill_(0)  # zero tz
            x_d_i_new = scale_new * (x_c_s @ R_new + delta_new) + t_new

            if inf_cfg.flag_relative_motion and inf_cfg.driving_option == "expression-friendly" and not flag_is_source_video and flag_is_driving_video:
                if i == 0:
                    x_d_0_new = x_d_i_new
                    motion_multiplier = calc_motion_multiplier(x_s, x_d_0_new)
                    # motion_multiplier *= inf_cfg.driving_multiplier
                x_d_diff = (x_d_i_new - x_d_0_new) * motion_multiplier
                x_d_i_new = x_d_diff + x_s

            # Algorithm 1:
            if not inf_cfg.flag_stitching and not inf_cfg.flag_eye_retargeting and not inf_cfg.flag_lip_retargeting:
                # without stitching or retargeting
                if flag_normalize_lip and lip_delta_before_animation is not None:
                    x_d_i_new += lip_delta_before_animation
                if flag_source_video_eye_retargeting and eye_delta_before_animation is not None:
                    x_d_i_new += eye_delta_before_animation
                else:
                    pass
            elif inf_cfg.flag_stitching and not inf_cfg.flag_eye_retargeting and not inf_cfg.flag_lip_retargeting:
                # with stitching and without retargeting
                if flag_normalize_lip and lip_delta_before_animation is not None:
                    x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new) + lip_delta_before_animation
                else:
                    x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new)
                if flag_source_video_eye_retargeting and eye_delta_before_animation is not None:
                    x_d_i_new += eye_delta_before_animation
            else:
                eyes_delta, lip_delta = None, None
                if inf_cfg.flag_eye_retargeting and source_lmk is not None:
                    c_d_eyes_i = c_d_eyes_lst[i]
                    combined_eye_ratio_tensor = self.live_portrait_wrapper.calc_combined_eye_ratio(c_d_eyes_i, source_lmk)
                    # ∆_eyes,i = R_eyes(x_s; c_s,eyes, c_d,eyes,i)
                    eyes_delta = self.live_portrait_wrapper.retarget_eye(x_s, combined_eye_ratio_tensor)
                if inf_cfg.flag_lip_retargeting and source_lmk is not None:
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

            x_d_i_new = x_s + (x_d_i_new - x_s) * inf_cfg.driving_multiplier
            out = self.live_portrait_wrapper.warp_decode(f_s, x_s, x_d_i_new)
            I_p_i = self.live_portrait_wrapper.parse_output(out['out'])[0]
            I_p_lst.append(I_p_i)

            if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:
                # TODO: the paste back procedure is slow, considering optimize it using multi-threading or GPU
                if flag_is_source_video:
                    I_p_pstbk = paste_back(I_p_i, source_M_c2o_lst[i], source_rgb_lst[i], mask_ori_float)
                else:
                    I_p_pstbk = paste_back(I_p_i, crop_info['M_c2o'], source_rgb_lst[0], mask_ori_float)
                I_p_pstbk_lst.append(I_p_pstbk)

        mkdir(args.output_dir)
        wfp_concat = None
        ######### build the final concatenation result #########
        # driving frame | source frame | generation
        if flag_is_source_video and flag_is_driving_video:
            frames_concatenated = concat_frames(driving_rgb_crop_256x256_lst, img_crop_256x256_lst, I_p_lst)
        elif flag_is_source_video and not flag_is_driving_video:
            if flag_load_from_template:
                frames_concatenated = concat_frames(driving_rgb_crop_256x256_lst, img_crop_256x256_lst, I_p_lst)
            else:
                frames_concatenated = concat_frames(driving_rgb_crop_256x256_lst*n_frames, img_crop_256x256_lst, I_p_lst)
        else:
            frames_concatenated = concat_frames(driving_rgb_crop_256x256_lst, [img_crop_256x256], I_p_lst)

        if flag_is_driving_video or (flag_is_source_video and not flag_is_driving_video):
            flag_source_has_audio = flag_is_source_video and has_audio_stream(args.source)
            flag_driving_has_audio = (not flag_load_from_template) and has_audio_stream(args.driving)

            wfp_concat = osp.join(args.output_dir, f'{basename(args.source)}--{basename(args.driving)}_concat.mp4')

            # NOTE: update output fps
            output_fps = source_fps if flag_is_source_video else output_fps
            images2video(frames_concatenated, wfp=wfp_concat, fps=output_fps)

            if flag_source_has_audio or flag_driving_has_audio:
                # final result with concatenation
                wfp_concat_with_audio = osp.join(args.output_dir, f'{basename(args.source)}--{basename(args.driving)}_concat_with_audio.mp4')
                audio_from_which_video = args.driving if ((flag_driving_has_audio and args.audio_priority == 'driving') or (not flag_source_has_audio)) else args.source
                log(f"Audio is selected from {audio_from_which_video}, concat mode")
                add_audio_to_video(wfp_concat, audio_from_which_video, wfp_concat_with_audio)
                os.replace(wfp_concat_with_audio, wfp_concat)
                log(f"Replace {wfp_concat_with_audio} with {wfp_concat}")

            # save the animated result
            wfp = osp.join(args.output_dir, f'{basename(args.source)}--{basename(args.driving)}.mp4')
            if I_p_pstbk_lst is not None and len(I_p_pstbk_lst) > 0:
                images2video(I_p_pstbk_lst, wfp=wfp, fps=output_fps)
            else:
                images2video(I_p_lst, wfp=wfp, fps=output_fps)

            ######### build the final result #########
            if flag_source_has_audio or flag_driving_has_audio:
                wfp_with_audio = osp.join(args.output_dir, f'{basename(args.source)}--{basename(args.driving)}_with_audio.mp4')
                audio_from_which_video = args.driving if ((flag_driving_has_audio and args.audio_priority == 'driving') or (not flag_source_has_audio)) else args.source
                log(f"Audio is selected from {audio_from_which_video}")
                add_audio_to_video(wfp, audio_from_which_video, wfp_with_audio)
                os.replace(wfp_with_audio, wfp)
                log(f"Replace {wfp_with_audio} with {wfp}")

            # final log
            if wfp_template not in (None, ''):
                log(f'Animated template: {wfp_template}, you can specify `-d` argument with this template path next time to avoid cropping video, motion making and protecting privacy.', style='bold green')
            log(f'Animated video: {wfp}')
            log(f'Animated video with concat: {wfp_concat}')
        else:
            wfp_concat = osp.join(args.output_dir, f'{basename(args.source)}--{basename(args.driving)}_concat.jpg')
            cv2.imwrite(wfp_concat, frames_concatenated[0][..., ::-1])
            wfp = osp.join(args.output_dir, f'{basename(args.source)}--{basename(args.driving)}.jpg')
            if I_p_pstbk_lst is not None and len(I_p_pstbk_lst) > 0:
                cv2.imwrite(wfp, I_p_pstbk_lst[0][..., ::-1])
            else:
                cv2.imwrite(wfp, frames_concatenated[0][..., ::-1])
            # final log
            log(f'Animated image: {wfp}')
            log(f'Animated image with concat: {wfp_concat}')

        return wfp, wfp_concat
