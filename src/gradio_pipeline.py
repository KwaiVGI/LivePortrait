# coding: utf-8

"""
Pipeline for gradio
"""

from .config.argument_config import ArgumentConfig
from .live_portrait_pipeline import LivePortraitPipeline
from .utils.io import load_img_online
from .utils.camera import get_rotation_matrix
from .utils.retargeting_utils import calc_eye_close_ratio, calc_lip_close_ratio
from .utils.rprint import rlog as log

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
        # for single image retargeting
        self.f_s_user = None
        self.x_c_s_info_user = None
        self.x_s_user = None
        self.source_lmk_user = None

    def execute_video(
        self,
        input_image_path,
        input_video_path,
        flag_relative_input,
        flag_do_crop_input,
        flag_remap_input
        ):
        """ for video driven potrait animation
        """
        args_user = {
            'source_image': input_image_path,
            'driving_info': input_video_path,
            'flag_relative': flag_relative_input,
            'flag_do_crop': flag_do_crop_input,
            'flag_pasteback': flag_remap_input
        }
        # update config from user input
        self.args = update_args(self.args, args_user)
        self.live_portrait_wrapper.update_config(self.args.__dict__)
        self.cropper.update_config(self.args.__dict__)
        # video driven animation
        video_path, video_path_concat = self.execute(self.args)
        return video_path, video_path_concat

    def execute_image(self, input_eye_ratio: float, input_lip_ratio: float):
        """ for single image retargeting
        """
        # ∆_eyes,i = R_eyes(x_s; c_s,eyes, c_d,eyes,i)
        combined_eye_ratio_tensor = self.live_portrait_wrapper.calc_combined_eye_ratio([[input_eye_ratio]], self.source_lmk_user)
        eyes_delta = self.live_portrait_wrapper.retarget_eye(self.x_s_user, combined_eye_ratio_tensor)
        # ∆_lip,i = R_lip(x_s; c_s,lip, c_d,lip,i)
        combined_lip_ratio_tensor = self.live_portrait_wrapper.calc_combined_lip_ratio([[input_lip_ratio]], self.source_lmk_user)
        lip_delta = self.live_portrait_wrapper.retarget_lip(self.x_s_user, combined_lip_ratio_tensor)
        num_kp = self.x_s_user.shape[1]
        # default: use x_s
        x_d_new = self.x_s_user + eyes_delta.reshape(-1, num_kp, 3) + lip_delta.reshape(-1, num_kp, 3)
        # D(W(f_s; x_s, x′_d))
        out = self.live_portrait_wrapper.warp_decode(self.f_s_user, self.x_s_user, x_d_new)
        out = self.live_portrait_wrapper.parse_output(out['out'])[0]
        return out

    def prepare_retargeting(self, input_image_path, flag_do_crop = True):
        """ for single image retargeting
        """
        inference_cfg = self.live_portrait_wrapper.cfg
        ######## process reference portrait ########
        img_rgb = load_img_online(input_image_path, mode='rgb', max_dim=1280, n=16)
        log(f"Load source image from {input_image_path}.")
        crop_info = self.cropper.crop_single_image(img_rgb)
        if flag_do_crop:
            I_s = self.live_portrait_wrapper.prepare_source(crop_info['img_crop_256x256'])
        else:
            I_s = self.live_portrait_wrapper.prepare_source(img_rgb)
        x_s_info = self.live_portrait_wrapper.get_kp_info(I_s)
        R_s = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
        ############################################

        # record global info for next time use
        self.f_s_user = self.live_portrait_wrapper.extract_feature_3d(I_s)
        self.x_s_user = self.live_portrait_wrapper.transform_keypoint(x_s_info)
        self.x_s_info_user = x_s_info
        self.source_lmk_user = crop_info['lmk_crop']

        # update slider
        eye_close_ratio = calc_eye_close_ratio(self.source_lmk_user[None])
        eye_close_ratio = float(eye_close_ratio.squeeze(0).mean())
        lip_close_ratio = calc_lip_close_ratio(self.source_lmk_user[None])
        lip_close_ratio = float(lip_close_ratio.squeeze(0).mean())

        return eye_close_ratio, lip_close_ratio
