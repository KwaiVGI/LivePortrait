# coding: utf-8

"""
Pipeline for gradio
"""

import os.path as osp
import os
import cv2
from rich.progress import track
import gradio as gr
import numpy as np
import torch

from .config.argument_config import ArgumentConfig
from .live_portrait_pipeline import LivePortraitPipeline
from .live_portrait_pipeline_animal import LivePortraitPipelineAnimal
from .utils.io import load_img_online, load_video, resize_to_limit
from .utils.filter import smooth
from .utils.rprint import rlog as log
from .utils.crop import prepare_paste_back, paste_back
from .utils.camera import get_rotation_matrix
from .utils.video import get_fps, has_audio_stream, concat_frames, images2video, add_audio_to_video
from .utils.helper import is_square_video, mkdir, dct2device, basename
from .utils.retargeting_utils import calc_eye_close_ratio, calc_lip_close_ratio
from skimage.exposure import match_histograms

def update_args(args, user_args):
    """update the args according to user inputs
    """
    for k, v in user_args.items():
        if hasattr(args, k):
            setattr(args, k, v)
    return args


class GradioPipeline(LivePortraitPipeline):
    """gradio for human
    """

    def __init__(self, inference_cfg, crop_cfg, args: ArgumentConfig):
        super().__init__(inference_cfg, crop_cfg)
        # self.live_portrait_wrapper = self.live_portrait_wrapper
        self.args = args

    @torch.no_grad()
    def update_delta_new_eyeball_direction(self, eyeball_direction_x, eyeball_direction_y, delta_new, **kwargs):
        if eyeball_direction_x > 0:
                delta_new[0, 11, 0] += eyeball_direction_x * 0.0007
                delta_new[0, 15, 0] += eyeball_direction_x * 0.001
        else:
            delta_new[0, 11, 0] += eyeball_direction_x * 0.001
            delta_new[0, 15, 0] += eyeball_direction_x * 0.0007

        delta_new[0, 11, 1] += eyeball_direction_y * -0.001
        delta_new[0, 15, 1] += eyeball_direction_y * -0.001
        blink = -eyeball_direction_y / 2.

        delta_new[0, 11, 1] += blink * -0.001
        delta_new[0, 13, 1] += blink * 0.0003
        delta_new[0, 15, 1] += blink * -0.001
        delta_new[0, 16, 1] += blink * 0.0003

        return delta_new

    @torch.no_grad()
    def update_delta_new_smile(self, smile, delta_new, **kwargs):
        delta_new[0, 20, 1] += smile * -0.01
        delta_new[0, 14, 1] += smile * -0.02
        delta_new[0, 17, 1] += smile * 0.0065
        delta_new[0, 17, 2] += smile * 0.003
        delta_new[0, 13, 1] += smile * -0.00275
        delta_new[0, 16, 1] += smile * -0.00275
        delta_new[0, 3, 1] += smile * -0.0035
        delta_new[0, 7, 1] += smile * -0.0035

        return delta_new

    @torch.no_grad()
    def update_delta_new_wink(self, wink, delta_new, **kwargs):
        delta_new[0, 11, 1] += wink * 0.001
        delta_new[0, 13, 1] += wink * -0.0003
        delta_new[0, 17, 0] += wink * 0.0003
        delta_new[0, 17, 1] += wink * 0.0003
        delta_new[0, 3, 1] += wink * -0.0003

        return delta_new

    @torch.no_grad()
    def update_delta_new_eyebrow(self, eyebrow, delta_new, **kwargs):
        if eyebrow > 0:
            delta_new[0, 1, 1] += eyebrow * 0.001
            delta_new[0, 2, 1] += eyebrow * -0.001
        else:
            delta_new[0, 1, 0] += eyebrow * -0.001
            delta_new[0, 2, 0] += eyebrow * 0.001
            delta_new[0, 1, 1] += eyebrow * 0.0003
            delta_new[0, 2, 1] += eyebrow * -0.0003
        return delta_new

    @torch.no_grad()
    def update_delta_new_lip_variation_zero(self, lip_variation_zero, delta_new, **kwargs):
        delta_new[0, 19, 0] += lip_variation_zero

        return delta_new

    @torch.no_grad()
    def update_delta_new_lip_variation_one(self, lip_variation_one, delta_new, **kwargs):
        delta_new[0, 14, 1] += lip_variation_one * 0.001
        delta_new[0, 3, 1] += lip_variation_one * -0.0005
        delta_new[0, 7, 1] += lip_variation_one * -0.0005
        delta_new[0, 17, 2] += lip_variation_one * -0.0005

        return delta_new

    @torch.no_grad()
    def update_delta_new_lip_variation_two(self, lip_variation_two, delta_new, **kwargs):
        delta_new[0, 20, 2] += lip_variation_two * -0.001
        delta_new[0, 20, 1] += lip_variation_two * -0.001
        delta_new[0, 14, 1] += lip_variation_two * -0.001

        return delta_new

    @torch.no_grad()
    def update_delta_new_lip_variation_three(self, lip_variation_three, delta_new, **kwargs):
        delta_new[0, 19, 1] += lip_variation_three * 0.001
        delta_new[0, 19, 2] += lip_variation_three * 0.0001
        delta_new[0, 17, 1] += lip_variation_three * -0.0001

        return delta_new

    @torch.no_grad()
    def update_delta_new_mov_x(self, mov_x, delta_new, **kwargs):
        delta_new[0, 5, 0] += mov_x

        return delta_new

    @torch.no_grad()
    def update_delta_new_mov_y(self, mov_y, delta_new, **kwargs):
        delta_new[0, 5, 1] += mov_y

        return delta_new

    @torch.no_grad()
    def execute_video(
        self,
        input_source_image_path=None,
        input_source_video_path=None,
        input_driving_video_path=None,
        input_driving_image_path=None,
        input_driving_video_pickle_path=None,
        flag_normalize_lip=False,
        flag_relative_input=True,
        flag_do_crop_input=True,
        flag_remap_input=True,
        flag_stitching_input=True,
        animation_region="all",
        driving_option_input="pose-friendly",
        driving_multiplier=1.0,
        flag_crop_driving_video_input=True,
        # flag_video_editing_head_rotation=False,
        scale=2.3,
        vx_ratio=0.0,
        vy_ratio=-0.125,
        scale_crop_driving_video=2.2,
        vx_ratio_crop_driving_video=0.0,
        vy_ratio_crop_driving_video=-0.1,
        driving_smooth_observation_variance=3e-7,
        tab_selection=None,
        v_tab_selection=None
    ):
        """ for video-driven portrait animation or video editing
        """
        if tab_selection == 'Image':
            input_source_path = input_source_image_path
        elif tab_selection == 'Video':
            input_source_path = input_source_video_path
        else:
            input_source_path = input_source_image_path

        if v_tab_selection == 'Video':
            input_driving_path = input_driving_video_path
        elif v_tab_selection == 'Image':
            input_driving_path = input_driving_image_path
        elif v_tab_selection == 'Pickle':
            input_driving_path = input_driving_video_pickle_path
        else:
            input_driving_path = input_driving_video_path

        if input_source_path is not None and input_driving_path is not None:
            if osp.exists(input_driving_path) and v_tab_selection == 'Video' and not flag_crop_driving_video_input and is_square_video(input_driving_path) is False:
                flag_crop_driving_video_input = True
                log("The driving video is not square, it will be cropped to square automatically.")
                gr.Info("The driving video is not square, it will be cropped to square automatically.", duration=2)

            args_user = {
                'source': input_source_path,
                'driving': input_driving_path,
                'flag_normalize_lip' : flag_normalize_lip,
                'flag_relative_motion': flag_relative_input,
                'flag_do_crop': flag_do_crop_input,
                'flag_pasteback': flag_remap_input,
                'flag_stitching': flag_stitching_input,
                'animation_region': animation_region,
                'driving_option': driving_option_input,
                'driving_multiplier': driving_multiplier,
                'flag_crop_driving_video': flag_crop_driving_video_input,
                'scale': scale,
                'vx_ratio': vx_ratio,
                'vy_ratio': vy_ratio,
                'scale_crop_driving_video': scale_crop_driving_video,
                'vx_ratio_crop_driving_video': vx_ratio_crop_driving_video,
                'vy_ratio_crop_driving_video': vy_ratio_crop_driving_video,
                'driving_smooth_observation_variance': driving_smooth_observation_variance,
            }
            # update config from user input
            self.args = update_args(self.args, args_user)
            self.live_portrait_wrapper.update_config(self.args.__dict__)
            self.cropper.update_config(self.args.__dict__)

            output_path, output_path_concat = self.execute(self.args)
            gr.Info("Run successfully!", duration=2)
            if output_path.endswith(".jpg"):
                return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), output_path, gr.update(visible=True), output_path_concat, gr.update(visible=True)
            else:
                return output_path, gr.update(visible=True), output_path_concat, gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        else:
            raise gr.Error("Please upload the source portrait or source video, and driving video 🤗🤗🤗", duration=5)

    @torch.no_grad()
    def execute_image_retargeting(
        self,
        input_eye_ratio: float,
        input_lip_ratio: float,
        input_head_pitch_variation: float,
        input_head_yaw_variation: float,
        input_head_roll_variation: float,
        mov_x: float,
        mov_y: float,
        mov_z: float,
        lip_variation_zero: float,
        lip_variation_one: float,
        lip_variation_two: float,
        lip_variation_three: float,
        smile: float,
        wink: float,
        eyebrow: float,
        eyeball_direction_x: float,
        eyeball_direction_y: float,
        input_image,
        retargeting_source_scale: float,
        flag_stitching_retargeting_input=True,
        flag_do_crop_input_retargeting_image=True):
        """ for single image retargeting
        """
        if input_head_pitch_variation is None or input_head_yaw_variation is None or input_head_roll_variation is None:
            raise gr.Error("Invalid relative pose input 💥!", duration=5)
        # disposable feature
        f_s_user, x_s_user, R_s_user, R_d_user, x_s_info, source_lmk_user, crop_M_c2o, mask_ori, img_rgb = \
            self.prepare_retargeting_image(
                input_image, input_head_pitch_variation, input_head_yaw_variation, input_head_roll_variation, retargeting_source_scale, flag_do_crop=flag_do_crop_input_retargeting_image)

        if input_eye_ratio is None or input_lip_ratio is None:
            raise gr.Error("Invalid ratio input 💥!", duration=5)
        else:
            device = self.live_portrait_wrapper.device
            # inference_cfg = self.live_portrait_wrapper.inference_cfg
            x_s_user = x_s_user.to(device)
            f_s_user = f_s_user.to(device)
            R_s_user = R_s_user.to(device)
            R_d_user = R_d_user.to(device)
            mov_x = torch.tensor(mov_x).to(device)
            mov_y = torch.tensor(mov_y).to(device)
            mov_z = torch.tensor(mov_z).to(device)
            eyeball_direction_x = torch.tensor(eyeball_direction_x).to(device)
            eyeball_direction_y = torch.tensor(eyeball_direction_y).to(device)
            smile = torch.tensor(smile).to(device)
            wink = torch.tensor(wink).to(device)
            eyebrow = torch.tensor(eyebrow).to(device)
            lip_variation_zero = torch.tensor(lip_variation_zero).to(device)
            lip_variation_one = torch.tensor(lip_variation_one).to(device)
            lip_variation_two = torch.tensor(lip_variation_two).to(device)
            lip_variation_three = torch.tensor(lip_variation_three).to(device)

            x_c_s = x_s_info['kp'].to(device)
            delta_new = x_s_info['exp'].to(device)
            scale_new = x_s_info['scale'].to(device)
            t_new = x_s_info['t'].to(device)
            R_d_new = (R_d_user @ R_s_user.permute(0, 2, 1)) @ R_s_user

            if eyeball_direction_x != 0 or eyeball_direction_y != 0:
                delta_new = self.update_delta_new_eyeball_direction(eyeball_direction_x, eyeball_direction_y, delta_new)
            if smile != 0:
                delta_new = self.update_delta_new_smile(smile, delta_new)
            if wink != 0:
                delta_new = self.update_delta_new_wink(wink, delta_new)
            if eyebrow != 0:
                delta_new = self.update_delta_new_eyebrow(eyebrow, delta_new)
            if lip_variation_zero != 0:
                delta_new = self.update_delta_new_lip_variation_zero(lip_variation_zero, delta_new)
            if lip_variation_one !=  0:
                delta_new = self.update_delta_new_lip_variation_one(lip_variation_one, delta_new)
            if lip_variation_two != 0:
                delta_new = self.update_delta_new_lip_variation_two(lip_variation_two, delta_new)
            if lip_variation_three != 0:
                delta_new = self.update_delta_new_lip_variation_three(lip_variation_three, delta_new)
            if mov_x != 0:
                delta_new = self.update_delta_new_mov_x(-mov_x, delta_new)
            if mov_y !=0 :
                delta_new = self.update_delta_new_mov_y(mov_y, delta_new)

            x_d_new = mov_z * scale_new * (x_c_s @ R_d_new + delta_new) + t_new
            eyes_delta, lip_delta = None, None
            if input_eye_ratio != self.source_eye_ratio:
                combined_eye_ratio_tensor = self.live_portrait_wrapper.calc_combined_eye_ratio([[float(input_eye_ratio)]], source_lmk_user)
                eyes_delta = self.live_portrait_wrapper.retarget_eye(x_s_user, combined_eye_ratio_tensor)
            if input_lip_ratio != self.source_lip_ratio:
                combined_lip_ratio_tensor = self.live_portrait_wrapper.calc_combined_lip_ratio([[float(input_lip_ratio)]], source_lmk_user)
                lip_delta = self.live_portrait_wrapper.retarget_lip(x_s_user, combined_lip_ratio_tensor)
                print(lip_delta)
            x_d_new = x_d_new + \
                    (eyes_delta if eyes_delta is not None else 0) + \
                    (lip_delta if lip_delta is not None else 0)

            if flag_stitching_retargeting_input:
                x_d_new = self.live_portrait_wrapper.stitching(x_s_user, x_d_new)
            out = self.live_portrait_wrapper.warp_decode(f_s_user, x_s_user, x_d_new)
            out = self.live_portrait_wrapper.parse_output(out['out'])[0]
            if flag_do_crop_input_retargeting_image:
                out_to_ori_blend = paste_back(out, crop_M_c2o, img_rgb, mask_ori)
            else:
                out_to_ori_blend = out
            return out, out_to_ori_blend

    @torch.no_grad()
    def prepare_retargeting_image(
        self,
        input_image,
        input_head_pitch_variation, input_head_yaw_variation, input_head_roll_variation,
        retargeting_source_scale,
        flag_do_crop=True):
        """ for single image retargeting
        """
        if input_image is not None:
            # gr.Info("Upload successfully!", duration=2)
            args_user = {'scale': retargeting_source_scale}
            self.args = update_args(self.args, args_user)
            self.cropper.update_config(self.args.__dict__)
            inference_cfg = self.live_portrait_wrapper.inference_cfg
            ######## process source portrait ########
            img_rgb = load_img_online(input_image, mode='rgb', max_dim=1280, n=2)
            if flag_do_crop:
                crop_info = self.cropper.crop_source_image(img_rgb, self.cropper.crop_cfg)
                I_s = self.live_portrait_wrapper.prepare_source(crop_info['img_crop_256x256'])
                source_lmk_user = crop_info['lmk_crop']
                crop_M_c2o = crop_info['M_c2o']
                mask_ori = prepare_paste_back(inference_cfg.mask_crop, crop_info['M_c2o'], dsize=(img_rgb.shape[1], img_rgb.shape[0]))
            else:
                I_s = self.live_portrait_wrapper.prepare_source(img_rgb)
                source_lmk_user = self.cropper.calc_lmk_from_cropped_image(img_rgb)
                crop_M_c2o = None
                mask_ori = None
            x_s_info = self.live_portrait_wrapper.get_kp_info(I_s)
            x_d_info_user_pitch = x_s_info['pitch'] + input_head_pitch_variation
            x_d_info_user_yaw = x_s_info['yaw'] + input_head_yaw_variation
            x_d_info_user_roll = x_s_info['roll'] + input_head_roll_variation
            R_s_user = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
            R_d_user = get_rotation_matrix(x_d_info_user_pitch, x_d_info_user_yaw, x_d_info_user_roll)
            ############################################
            f_s_user = self.live_portrait_wrapper.extract_feature_3d(I_s)
            x_s_user = self.live_portrait_wrapper.transform_keypoint(x_s_info)
            return f_s_user, x_s_user, R_s_user, R_d_user, x_s_info, source_lmk_user, crop_M_c2o, mask_ori, img_rgb
        else:
            raise gr.Error("Please upload a source portrait as the retargeting input 🤗🤗🤗", duration=5)

    @torch.no_grad()
    def init_retargeting_image(self, retargeting_source_scale: float, source_eye_ratio: float, source_lip_ratio:float, input_image = None):
        """ initialize the retargeting slider
        """
        if input_image != None:
            args_user = {'scale': retargeting_source_scale}
            self.args = update_args(self.args, args_user)
            self.cropper.update_config(self.args.__dict__)
            # inference_cfg = self.live_portrait_wrapper.inference_cfg
            ######## process source portrait ########
            img_rgb = load_img_online(input_image, mode='rgb', max_dim=1280, n=16)
            log(f"Load source image from {input_image}.")
            crop_info = self.cropper.crop_source_image(img_rgb, self.cropper.crop_cfg)
            if crop_info is None:
                raise gr.Error("Source portrait NO face detected", duration=2)
            source_eye_ratio = calc_eye_close_ratio(crop_info['lmk_crop'][None])
            source_lip_ratio = calc_lip_close_ratio(crop_info['lmk_crop'][None])
            self.source_eye_ratio = round(float(source_eye_ratio.mean()), 2)
            self.source_lip_ratio = round(float(source_lip_ratio[0][0]), 2)
            log("Calculating eyes-open and lip-open ratios successfully!")
            return self.source_eye_ratio, self.source_lip_ratio
        else:
            return source_eye_ratio, source_lip_ratio

    @torch.no_grad()
    def execute_video_retargeting(self, input_lip_ratio: float, input_video, retargeting_source_scale: float, driving_smooth_observation_variance_retargeting: float, video_retargeting_silence=False, flag_do_crop_input_retargeting_video=True):
        """ retargeting the lip-open ratio of each source frame
        """
        # disposable feature
        device = self.live_portrait_wrapper.device

        if not video_retargeting_silence:
            f_s_user_lst, x_s_user_lst, source_lmk_crop_lst, source_M_c2o_lst, mask_ori_lst, source_rgb_lst, img_crop_256x256_lst, lip_delta_retargeting_lst_smooth, source_fps, n_frames = \
                self.prepare_retargeting_video(input_video, retargeting_source_scale, device, input_lip_ratio, driving_smooth_observation_variance_retargeting, flag_do_crop=flag_do_crop_input_retargeting_video)
            if input_lip_ratio is None:
                raise gr.Error("Invalid ratio input 💥!", duration=5)
            else:
                inference_cfg = self.live_portrait_wrapper.inference_cfg

                I_p_pstbk_lst = None
                if flag_do_crop_input_retargeting_video:
                    I_p_pstbk_lst = []
                I_p_lst = []
                for i in track(range(n_frames), description='Retargeting video...', total=n_frames):
                    x_s_user_i = x_s_user_lst[i].to(device)
                    f_s_user_i = f_s_user_lst[i].to(device)

                    lip_delta_retargeting = lip_delta_retargeting_lst_smooth[i]
                    x_d_i_new = x_s_user_i + lip_delta_retargeting
                    x_d_i_new = self.live_portrait_wrapper.stitching(x_s_user_i, x_d_i_new)
                    out = self.live_portrait_wrapper.warp_decode(f_s_user_i, x_s_user_i, x_d_i_new)
                    I_p_i = self.live_portrait_wrapper.parse_output(out['out'])[0]
                    I_p_lst.append(I_p_i)

                    if flag_do_crop_input_retargeting_video:
                        I_p_i = match_histograms(I_p_i,img_crop_256x256_lst[i])
                        I_p_i = np.clip(I_p_i,0,255).astype(np.uint8)
                        I_p_pstbk = paste_back(I_p_i, source_M_c2o_lst[i], source_rgb_lst[i], mask_ori_lst[i])
                        I_p_pstbk_lst.append(I_p_pstbk)
        else:
            inference_cfg = self.live_portrait_wrapper.inference_cfg
            f_s_user_lst, x_s_user_lst, x_d_i_new_lst, source_M_c2o_lst, mask_ori_lst, source_rgb_lst, img_crop_256x256_lst, source_fps, n_frames = \
                self.prepare_video_lip_silence(input_video, device, flag_do_crop=flag_do_crop_input_retargeting_video)

            I_p_pstbk_lst = None
            if flag_do_crop_input_retargeting_video:
                I_p_pstbk_lst = []
            I_p_lst = []
            for i in track(range(n_frames), description='Silencing lip...', total=n_frames):
                x_s_user_i = x_s_user_lst[i].to(device)
                f_s_user_i = f_s_user_lst[i].to(device)
                x_d_i_new = x_d_i_new_lst[i]
                x_d_i_new = self.live_portrait_wrapper.stitching(x_s_user_i, x_d_i_new)
                out = self.live_portrait_wrapper.warp_decode(f_s_user_i, x_s_user_i, x_d_i_new)
                I_p_i = self.live_portrait_wrapper.parse_output(out['out'])[0]
                I_p_lst.append(I_p_i)

                if flag_do_crop_input_retargeting_video:
                    I_p_i = match_histograms(I_p_i,img_crop_256x256_lst[i])
                    I_p_i = np.clip(I_p_i,0,255).astype(np.uint8)
                    I_p_pstbk = paste_back(I_p_i, source_M_c2o_lst[i], source_rgb_lst[i], mask_ori_lst[i])
                    I_p_pstbk_lst.append(I_p_pstbk)

        mkdir(self.args.output_dir)
        flag_source_has_audio = has_audio_stream(input_video)

        ######### build the final concatenation result #########
        # source frame | generation
        frames_concatenated = concat_frames(driving_image_lst=None, source_image_lst=img_crop_256x256_lst, I_p_lst=I_p_lst)
        wfp_concat = osp.join(self.args.output_dir, f'{basename(input_video)}_retargeting_concat.mp4')
        images2video(frames_concatenated, wfp=wfp_concat, fps=source_fps)

        if flag_source_has_audio:
            # final result with concatenation
            wfp_concat_with_audio = osp.join(self.args.output_dir, f'{basename(input_video)}_retargeting_concat_with_audio.mp4')
            add_audio_to_video(wfp_concat, input_video, wfp_concat_with_audio)
            os.replace(wfp_concat_with_audio, wfp_concat)
            log(f"Replace {wfp_concat_with_audio} with {wfp_concat}")

        # save the animated result
        wfp = osp.join(self.args.output_dir, f'{basename(input_video)}_retargeting.mp4')
        if I_p_pstbk_lst is not None and len(I_p_pstbk_lst) > 0:
            images2video(I_p_pstbk_lst, wfp=wfp, fps=source_fps)
        else:
            images2video(I_p_lst, wfp=wfp, fps=source_fps)

        ######### build the final result #########
        if flag_source_has_audio:
            wfp_with_audio = osp.join(self.args.output_dir, f'{basename(input_video)}_retargeting_with_audio.mp4')
            add_audio_to_video(wfp, input_video, wfp_with_audio)
            os.replace(wfp_with_audio, wfp)
            log(f"Replace {wfp_with_audio} with {wfp}")
        gr.Info("Run successfully!", duration=2)
        return wfp_concat, wfp

    @torch.no_grad()
    def prepare_retargeting_video(self, input_video, retargeting_source_scale, device, input_lip_ratio, driving_smooth_observation_variance_retargeting, flag_do_crop=True):
        """ for video retargeting
        """
        if input_video is not None:
            # gr.Info("Upload successfully!", duration=2)
            args_user = {'scale': retargeting_source_scale}
            self.args = update_args(self.args, args_user)
            self.cropper.update_config(self.args.__dict__)
            inference_cfg = self.live_portrait_wrapper.inference_cfg
            ######## process source video ########
            source_rgb_lst = load_video(input_video)
            source_rgb_lst = [resize_to_limit(img, inference_cfg.source_max_dim, inference_cfg.source_division) for img in source_rgb_lst]
            source_fps = int(get_fps(input_video))
            n_frames = len(source_rgb_lst)
            log(f"Load source video from {input_video}. FPS is {source_fps}")

            if flag_do_crop:
                ret_s = self.cropper.crop_source_video(source_rgb_lst, self.cropper.crop_cfg)
                log(f'Source video is cropped, {len(ret_s["frame_crop_lst"])} frames are processed.')
                if len(ret_s["frame_crop_lst"]) != n_frames:
                    n_frames = min(len(source_rgb_lst), len(ret_s["frame_crop_lst"]))
                img_crop_256x256_lst, source_lmk_crop_lst, source_M_c2o_lst = ret_s['frame_crop_lst'], ret_s['lmk_crop_lst'], ret_s['M_c2o_lst']
                mask_ori_lst = [prepare_paste_back(inference_cfg.mask_crop, source_M_c2o, dsize=(source_rgb_lst[0].shape[1], source_rgb_lst[0].shape[0])) for source_M_c2o in source_M_c2o_lst]
            else:
                source_lmk_crop_lst = self.cropper.calc_lmks_from_cropped_video(source_rgb_lst)
                img_crop_256x256_lst = [cv2.resize(_, (256, 256)) for _ in source_rgb_lst]  # force to resize to 256x256
                source_M_c2o_lst, mask_ori_lst = None, None

            c_s_eyes_lst, c_s_lip_lst = self.live_portrait_wrapper.calc_ratio(source_lmk_crop_lst)
            # save the motion template
            I_s_lst = self.live_portrait_wrapper.prepare_videos(img_crop_256x256_lst)
            source_template_dct = self.make_motion_template(I_s_lst, c_s_eyes_lst, c_s_lip_lst, output_fps=source_fps)

            c_d_lip_retargeting = [input_lip_ratio]
            f_s_user_lst, x_s_user_lst, lip_delta_retargeting_lst = [], [], []
            for i in track(range(n_frames), description='Preparing retargeting video...', total=n_frames):
                x_s_info = source_template_dct['motion'][i]
                x_s_info = dct2device(x_s_info, device)
                x_s_user = x_s_info['x_s']

                source_lmk = source_lmk_crop_lst[i]
                img_crop_256x256 = img_crop_256x256_lst[i]
                I_s = I_s_lst[i]
                f_s_user = self.live_portrait_wrapper.extract_feature_3d(I_s)

                combined_lip_ratio_tensor_retargeting = self.live_portrait_wrapper.calc_combined_lip_ratio(c_d_lip_retargeting, source_lmk)
                lip_delta_retargeting = self.live_portrait_wrapper.retarget_lip(x_s_user, combined_lip_ratio_tensor_retargeting)
                f_s_user_lst.append(f_s_user); x_s_user_lst.append(x_s_user); lip_delta_retargeting_lst.append(lip_delta_retargeting.cpu().numpy().astype(np.float32))
            lip_delta_retargeting_lst_smooth = smooth(lip_delta_retargeting_lst, lip_delta_retargeting_lst[0].shape, device, driving_smooth_observation_variance_retargeting)

            return f_s_user_lst, x_s_user_lst, source_lmk_crop_lst, source_M_c2o_lst, mask_ori_lst, source_rgb_lst, img_crop_256x256_lst, lip_delta_retargeting_lst_smooth, source_fps, n_frames
        else:
            # when press the clear button, go here
            raise gr.Error("Please upload a source video as the retargeting input 🤗🤗🤗", duration=5)

    @torch.no_grad()
    def prepare_video_lip_silence(self, input_video, device, flag_do_crop=True):
        """ for keeping lips in the source video silent
        """
        if input_video is not None:
            inference_cfg = self.live_portrait_wrapper.inference_cfg
            ######## process source video ########
            source_rgb_lst = load_video(input_video)
            source_rgb_lst = [resize_to_limit(img, inference_cfg.source_max_dim, inference_cfg.source_division) for img in source_rgb_lst]
            source_fps = int(get_fps(input_video))
            n_frames = len(source_rgb_lst)
            log(f"Load source video from {input_video}. FPS is {source_fps}")

            if flag_do_crop:
                ret_s = self.cropper.crop_source_video(source_rgb_lst, self.cropper.crop_cfg)
                log(f'Source video is cropped, {len(ret_s["frame_crop_lst"])} frames are processed.')
                if len(ret_s["frame_crop_lst"]) != n_frames:
                    n_frames = min(len(source_rgb_lst), len(ret_s["frame_crop_lst"]))
                img_crop_256x256_lst, source_lmk_crop_lst, source_M_c2o_lst = ret_s['frame_crop_lst'], ret_s['lmk_crop_lst'], ret_s['M_c2o_lst']
                mask_ori_lst = [prepare_paste_back(inference_cfg.mask_crop, source_M_c2o, dsize=(source_rgb_lst[0].shape[1], source_rgb_lst[0].shape[0])) for source_M_c2o in source_M_c2o_lst]
            else:
                source_lmk_crop_lst = self.cropper.calc_lmks_from_cropped_video(source_rgb_lst)
                img_crop_256x256_lst = [cv2.resize(_, (256, 256)) for _ in source_rgb_lst]  # force to resize to 256x256
                source_M_c2o_lst, mask_ori_lst = None, None

            c_s_eyes_lst, c_s_lip_lst = self.live_portrait_wrapper.calc_ratio(source_lmk_crop_lst)
            # save the motion template
            I_s_lst = self.live_portrait_wrapper.prepare_videos(img_crop_256x256_lst)
            source_template_dct = self.make_motion_template(I_s_lst, c_s_eyes_lst, c_s_lip_lst, output_fps=source_fps)

            f_s_user_lst, x_s_user_lst, x_d_i_new_lst = [], [], []
            for i in track(range(n_frames), description='Preparing silencing lip...', total=n_frames):
                x_s_info = source_template_dct['motion'][i]
                x_s_info = dct2device(x_s_info, device)
                scale_s = x_s_info['scale']
                x_s_user = x_s_info['x_s']
                x_c_s = x_s_info['kp']
                R_s = x_s_info['R']
                t_s = x_s_info['t']
                delta_new = torch.zeros_like(x_s_info['exp']) + torch.from_numpy(inference_cfg.lip_array).to(dtype=torch.float32, device=device)
                for eyes_idx in [11, 13, 15, 16, 18]:
                    delta_new[:, eyes_idx, :] = x_s_info['exp'][:, eyes_idx, :]
                source_lmk = source_lmk_crop_lst[i]
                img_crop_256x256 = img_crop_256x256_lst[i]
                I_s = I_s_lst[i]
                f_s_user = self.live_portrait_wrapper.extract_feature_3d(I_s)
                x_d_i_new = scale_s * (x_c_s @ R_s + delta_new) + t_s
                f_s_user_lst.append(f_s_user); x_s_user_lst.append(x_s_user); x_d_i_new_lst.append(x_d_i_new)
            return f_s_user_lst, x_s_user_lst, x_d_i_new_lst, source_M_c2o_lst, mask_ori_lst, source_rgb_lst, img_crop_256x256_lst, source_fps, n_frames
        else:
            # when press the clear button, go here
            raise gr.Error("Please upload a source video as the input 🤗🤗🤗", duration=5)

class GradioPipelineAnimal(LivePortraitPipelineAnimal):
    """gradio for animal
    """
    def __init__(self, inference_cfg, crop_cfg, args: ArgumentConfig):
        inference_cfg.flag_crop_driving_video = True # ensure the face_analysis_wrapper is enabled
        super().__init__(inference_cfg, crop_cfg)
        # self.live_portrait_wrapper_animal = self.live_portrait_wrapper_animal
        self.args = args

    @torch.no_grad()
    def execute_video(
        self,
        input_source_image_path=None,
        input_driving_video_path=None,
        input_driving_video_pickle_path=None,
        flag_do_crop_input=False,
        flag_remap_input=False,
        driving_multiplier=1.0,
        flag_stitching=False,
        flag_crop_driving_video_input=False,
        scale=2.3,
        vx_ratio=0.0,
        vy_ratio=-0.125,
        scale_crop_driving_video=2.2,
        vx_ratio_crop_driving_video=0.0,
        vy_ratio_crop_driving_video=-0.1,
        tab_selection=None,
    ):
        """ for video-driven potrait animation
        """
        input_source_path = input_source_image_path

        if tab_selection == 'Video':
            input_driving_path = input_driving_video_path
        elif tab_selection == 'Pickle':
            input_driving_path = input_driving_video_pickle_path
        else:
            input_driving_path = input_driving_video_pickle_path

        if input_source_path is not None and input_driving_path is not None:
            if osp.exists(input_driving_path) and tab_selection == 'Video' and is_square_video(input_driving_path) is False:
                flag_crop_driving_video_input = True
                log("The driving video is not square, it will be cropped to square automatically.")
                gr.Info("The driving video is not square, it will be cropped to square automatically.", duration=2)

            args_user = {
                'source': input_source_path,
                'driving': input_driving_path,
                'flag_do_crop': flag_do_crop_input,
                'flag_pasteback': flag_remap_input,
                'driving_multiplier': driving_multiplier,
                'flag_stitching': flag_stitching,
                'flag_crop_driving_video': flag_crop_driving_video_input,
                'scale': scale,
                'vx_ratio': vx_ratio,
                'vy_ratio': vy_ratio,
                'scale_crop_driving_video': scale_crop_driving_video,
                'vx_ratio_crop_driving_video': vx_ratio_crop_driving_video,
                'vy_ratio_crop_driving_video': vy_ratio_crop_driving_video,
            }
            # update config from user input
            self.args = update_args(self.args, args_user)
            self.live_portrait_wrapper_animal.update_config(self.args.__dict__)
            self.cropper.update_config(self.args.__dict__)
            # video driven animation
            video_path, video_path_concat, video_gif_path = self.execute(self.args)
            gr.Info("Run successfully!", duration=2)
            return video_path, video_path_concat, video_gif_path
        else:
            raise gr.Error("Please upload the source animal image, and driving video 🤗🤗🤗", duration=5)
