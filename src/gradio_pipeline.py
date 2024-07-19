# coding: utf-8

"""
Pipeline for gradio
"""
import gradio as gr

from .config.argument_config import ArgumentConfig
from .live_portrait_pipeline import LivePortraitPipeline
from .utils.io import load_img_online
from .utils.rprint import rlog as log
from .utils.crop import prepare_paste_back, paste_back
from .utils.camera import get_rotation_matrix
from .utils.helper import is_square_video


def update_args(args, user_args):
    """update the args according to user inputs
    """
    for k, v in user_args.items():
        if hasattr(args, k):
            setattr(args, k, v)
    return args


class GradioPipeline(LivePortraitPipeline):

    def __init__(self, inference_cfg, crop_cfg, args: ArgumentConfig):
        super().__init__(inference_cfg, crop_cfg)
        # self.live_portrait_wrapper = self.live_portrait_wrapper
        self.args = args

    def execute_video(
        self,
        input_source_image_path=None,
        input_source_video_path=None,
        input_driving_video_path=None,
        flag_relative_input=True,
        flag_do_crop_input=True,
        flag_remap_input=True,
        flag_crop_driving_video_input=True,
        flag_video_editing_head_rotation=False,
        scale=2.3,
        vx_ratio=0.0,
        vy_ratio=-0.125,
        scale_crop_driving_video=2.2,
        vx_ratio_crop_driving_video=0.0,
        vy_ratio_crop_driving_video=-0.1,
        driving_smooth_observation_variance=3e-6,
        tab_selection=None,
    ):
        """ for video-driven potrait animation or video editing
        """
        if tab_selection == 'Image':
            input_source_path = input_source_image_path
        elif tab_selection == 'Video':
            input_source_path = input_source_video_path
            if is_square_video(input_source_path) is False:
                flag_crop_driving_video_input = True
                gr.Info("The source video is not square, the driving video will be cropped to square automatically.", duration=2)
        else:
            input_source_path = input_source_image_path

        if input_source_path is not None and input_driving_video_path is not None:
            args_user = {
                'source': input_source_path,
                'driving': input_driving_video_path,
                'flag_relative_motion': flag_relative_input,
                'flag_do_crop': flag_do_crop_input,
                'flag_pasteback': flag_remap_input,
                'flag_crop_driving_video': flag_crop_driving_video_input,
                'flag_video_editing_head_rotation': flag_video_editing_head_rotation,
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
            # video driven animation
            video_path, video_path_concat = self.execute(self.args)
            gr.Info("Run successfully!", duration=2)
            return video_path, video_path_concat,
        else:
            raise gr.Error("Please upload the source portrait or source video, and driving video 🤗🤗🤗", duration=5)

    def execute_image(self, input_eye_ratio: float, input_lip_ratio: float, input_image, flag_do_crop=True):
        """ for single image retargeting
        """
        # disposable feature
        f_s_user, x_s_user, source_lmk_user, crop_M_c2o, mask_ori, img_rgb = \
            self.prepare_retargeting(input_image, flag_do_crop)

        if input_eye_ratio is None or input_lip_ratio is None:
            raise gr.Error("Invalid ratio input 💥!", duration=5)
        else:
            inference_cfg = self.live_portrait_wrapper.inference_cfg
            x_s_user = x_s_user.to(self.live_portrait_wrapper.device)
            f_s_user = f_s_user.to(self.live_portrait_wrapper.device)
            # ∆_eyes,i = R_eyes(x_s; c_s,eyes, c_d,eyes,i)
            combined_eye_ratio_tensor = self.live_portrait_wrapper.calc_combined_eye_ratio([[input_eye_ratio]], source_lmk_user)
            eyes_delta = self.live_portrait_wrapper.retarget_eye(x_s_user, combined_eye_ratio_tensor)
            # ∆_lip,i = R_lip(x_s; c_s,lip, c_d,lip,i)
            combined_lip_ratio_tensor = self.live_portrait_wrapper.calc_combined_lip_ratio([[input_lip_ratio]], source_lmk_user)
            lip_delta = self.live_portrait_wrapper.retarget_lip(x_s_user, combined_lip_ratio_tensor)
            # default: use x_s
            x_d_new = x_s_user + eyes_delta + lip_delta
            # D(W(f_s; x_s, x′_d))
            out = self.live_portrait_wrapper.warp_decode(f_s_user, x_s_user, x_d_new)
            out = self.live_portrait_wrapper.parse_output(out['out'])[0]
            out_to_ori_blend = paste_back(out, crop_M_c2o, img_rgb, mask_ori)
            gr.Info("Run successfully!", duration=2)
            return out, out_to_ori_blend

    def prepare_retargeting(self, input_image, flag_do_crop=True):
        """ for single image retargeting
        """
        if input_image is not None:
            # gr.Info("Upload successfully!", duration=2)
            inference_cfg = self.live_portrait_wrapper.inference_cfg
            ######## process source portrait ########
            img_rgb = load_img_online(input_image, mode='rgb', max_dim=1280, n=16)
            log(f"Load source image from {input_image}.")
            crop_info = self.cropper.crop_source_image(img_rgb, self.cropper.crop_cfg)
            if flag_do_crop:
                I_s = self.live_portrait_wrapper.prepare_source(crop_info['img_crop_256x256'])
            else:
                I_s = self.live_portrait_wrapper.prepare_source(img_rgb)
            x_s_info = self.live_portrait_wrapper.get_kp_info(I_s)
            R_s = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
            ############################################
            f_s_user = self.live_portrait_wrapper.extract_feature_3d(I_s)
            x_s_user = self.live_portrait_wrapper.transform_keypoint(x_s_info)
            source_lmk_user = crop_info['lmk_crop']
            crop_M_c2o = crop_info['M_c2o']
            mask_ori = prepare_paste_back(inference_cfg.mask_crop, crop_info['M_c2o'], dsize=(img_rgb.shape[1], img_rgb.shape[0]))
            return f_s_user, x_s_user, source_lmk_user, crop_M_c2o, mask_ori, img_rgb
        else:
            # when press the clear button, go here
            raise gr.Error("Please upload a source portrait as the retargeting input 🤗🤗🤗", duration=5)
