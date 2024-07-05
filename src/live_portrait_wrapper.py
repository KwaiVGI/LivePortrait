# coding: utf-8

"""
Wrapper for LivePortrait core functions
"""

import os.path as osp
import numpy as np
import cv2
import torch
import yaml

from src.utils.timer import Timer
from src.utils.helper import load_model, concat_feat
from src.utils.camera import headpose_pred_to_degree, get_rotation_matrix
from .utils.retargeting_utils import calc_eye_close_ratio, calc_lip_close_ratio
from src.config.inference_config import InferenceConfig
from src.utils.rprint import rlog as log


class LivePortraitWrapper(object):

    def __init__(self, cfg: InferenceConfig):

        model_config = yaml.load(open(cfg.models_config, 'r'), Loader=yaml.SafeLoader)

        # init F
        self.appearance_feature_extractor = load_model(cfg.checkpoint_F, model_config, cfg.device_id, 'appearance_feature_extractor')
        log(f'Load appearance_feature_extractor done.')
        # init M
        self.motion_extractor = load_model(cfg.checkpoint_M, model_config, cfg.device_id, 'motion_extractor')
        log(f'Load motion_extractor done.')
        # init W
        self.warping_module = load_model(cfg.checkpoint_W, model_config, cfg.device_id, 'warping_module')
        log(f'Load warping_module done.')
        # init G
        self.spade_generator = load_model(cfg.checkpoint_G, model_config, cfg.device_id, 'spade_generator')
        log(f'Load spade_generator done.')
        # init S and R
        if cfg.checkpoint_S is not None and osp.exists(cfg.checkpoint_S):
            self.stitching_retargeting_module = load_model(cfg.checkpoint_S, model_config, cfg.device_id, 'stitching_retargeting_module')
            log(f'Load stitching_retargeting_module done.')
        else:
            self.stitching_retargeting_module = None

        self.cfg = cfg
        self.device_id = cfg.device_id
        self.timer = Timer()

    def update_config(self, user_args):
        for k, v in user_args.items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)

    def prepare_source(self, img: np.ndarray) -> torch.Tensor:
        """ construct the input as standard
        img: HxWx3, uint8, 256x256
        """
        h, w = img.shape[:2]
        if h != self.cfg.input_shape[0] or w != self.cfg.input_shape[1]:
            x = cv2.resize(img, (self.cfg.input_shape[0], self.cfg.input_shape[1]))
        else:
            x = img.copy()

        if x.ndim == 3:
            x = x[np.newaxis].astype(np.float32) / 255.  # HxWx3 -> 1xHxWx3, normalized to 0~1
        elif x.ndim == 4:
            x = x.astype(np.float32) / 255.  # BxHxWx3, normalized to 0~1
        else:
            raise ValueError(f'img ndim should be 3 or 4: {x.ndim}')
        x = np.clip(x, 0, 1)  # clip to 0~1
        x = torch.from_numpy(x).permute(0, 3, 1, 2)  # 1xHxWx3 -> 1x3xHxW
        x = x.cuda(self.device_id)
        return x

    def prepare_driving_videos(self, imgs) -> torch.Tensor:
        """ construct the input as standard
        imgs: NxBxHxWx3, uint8
        """
        if isinstance(imgs, list):
            _imgs = np.array(imgs)[..., np.newaxis]  # TxHxWx3x1
        elif isinstance(imgs, np.ndarray):
            _imgs = imgs
        else:
            raise ValueError(f'imgs type error: {type(imgs)}')

        y = _imgs.astype(np.float32) / 255.
        y = np.clip(y, 0, 1)  # clip to 0~1
        y = torch.from_numpy(y).permute(0, 4, 3, 1, 2)  # TxHxWx3x1 -> Tx1x3xHxW
        y = y.cuda(self.device_id)

        return y

    def extract_feature_3d(self, x: torch.Tensor) -> torch.Tensor:
        """ get the appearance feature of the image by F
        x: Bx3xHxW, normalized to 0~1
        """
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.cfg.flag_use_half_precision):
                feature_3d = self.appearance_feature_extractor(x)

        return feature_3d.float()

    def get_kp_info(self, x: torch.Tensor, **kwargs) -> dict:
        """ get the implicit keypoint information
        x: Bx3xHxW, normalized to 0~1
        flag_refine_info: whether to trandform the pose to degrees and the dimention of the reshape
        return: A dict contains keys: 'pitch', 'yaw', 'roll', 't', 'exp', 'scale', 'kp'
        """
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.cfg.flag_use_half_precision):
                kp_info = self.motion_extractor(x)

            if self.cfg.flag_use_half_precision:
                # float the dict
                for k, v in kp_info.items():
                    if isinstance(v, torch.Tensor):
                        kp_info[k] = v.float()

        flag_refine_info: bool = kwargs.get('flag_refine_info', True)
        if flag_refine_info:
            bs = kp_info['kp'].shape[0]
            kp_info['pitch'] = headpose_pred_to_degree(kp_info['pitch'])[:, None]  # Bx1
            kp_info['yaw'] = headpose_pred_to_degree(kp_info['yaw'])[:, None]  # Bx1
            kp_info['roll'] = headpose_pred_to_degree(kp_info['roll'])[:, None]  # Bx1
            kp_info['kp'] = kp_info['kp'].reshape(bs, -1, 3)  # BxNx3
            kp_info['exp'] = kp_info['exp'].reshape(bs, -1, 3)  # BxNx3

        return kp_info

    def get_pose_dct(self, kp_info: dict) -> dict:
        pose_dct = dict(
            pitch=headpose_pred_to_degree(kp_info['pitch']).item(),
            yaw=headpose_pred_to_degree(kp_info['yaw']).item(),
            roll=headpose_pred_to_degree(kp_info['roll']).item(),
        )
        return pose_dct

    def get_fs_and_kp_info(self, source_prepared, driving_first_frame):

        # get the canonical keypoints of source image by M
        source_kp_info = self.get_kp_info(source_prepared, flag_refine_info=True)
        source_rotation = get_rotation_matrix(source_kp_info['pitch'], source_kp_info['yaw'], source_kp_info['roll'])

        # get the canonical keypoints of first driving frame by M
        driving_first_frame_kp_info = self.get_kp_info(driving_first_frame, flag_refine_info=True)
        driving_first_frame_rotation = get_rotation_matrix(
            driving_first_frame_kp_info['pitch'],
            driving_first_frame_kp_info['yaw'],
            driving_first_frame_kp_info['roll']
        )

        # get feature volume by F
        source_feature_3d = self.extract_feature_3d(source_prepared)

        return source_kp_info, source_rotation, source_feature_3d, driving_first_frame_kp_info, driving_first_frame_rotation

    def transform_keypoint(self, kp_info: dict):
        """
        transform the implicit keypoints with the pose, shift, and expression deformation
        kp: BxNx3
        """
        kp = kp_info['kp']    # (bs, k, 3)
        pitch, yaw, roll = kp_info['pitch'], kp_info['yaw'], kp_info['roll']

        t, exp = kp_info['t'], kp_info['exp']
        scale = kp_info['scale']

        pitch = headpose_pred_to_degree(pitch)
        yaw = headpose_pred_to_degree(yaw)
        roll = headpose_pred_to_degree(roll)

        bs = kp.shape[0]
        if kp.ndim == 2:
            num_kp = kp.shape[1] // 3  # Bx(num_kpx3)
        else:
            num_kp = kp.shape[1]  # Bxnum_kpx3

        rot_mat = get_rotation_matrix(pitch, yaw, roll)    # (bs, 3, 3)

        # Eqn.2: s * (R * x_c,s + exp) + t
        kp_transformed = kp.view(bs, num_kp, 3) @ rot_mat + exp.view(bs, num_kp, 3)
        kp_transformed *= scale[..., None]  # (bs, k, 3) * (bs, 1, 1) = (bs, k, 3)
        kp_transformed[:, :, 0:2] += t[:, None, 0:2]  # remove z, only apply tx ty

        return kp_transformed

    def retarget_eye(self, kp_source: torch.Tensor, eye_close_ratio: torch.Tensor) -> torch.Tensor:
        """
        kp_source: BxNx3
        eye_close_ratio: Bx3
        Return: Bx(3*num_kp+2)
        """
        feat_eye = concat_feat(kp_source, eye_close_ratio)

        with torch.no_grad():
            delta = self.stitching_retargeting_module['eye'](feat_eye)

        return delta

    def retarget_lip(self, kp_source: torch.Tensor, lip_close_ratio: torch.Tensor) -> torch.Tensor:
        """
        kp_source: BxNx3
        lip_close_ratio: Bx2
        """
        feat_lip = concat_feat(kp_source, lip_close_ratio)

        with torch.no_grad():
            delta = self.stitching_retargeting_module['lip'](feat_lip)

        return delta

    def stitch(self, kp_source: torch.Tensor, kp_driving: torch.Tensor) -> torch.Tensor:
        """
        kp_source: BxNx3
        kp_driving: BxNx3
        Return: Bx(3*num_kp+2)
        """
        feat_stiching = concat_feat(kp_source, kp_driving)

        with torch.no_grad():
            delta = self.stitching_retargeting_module['stitching'](feat_stiching)

        return delta

    def stitching(self, kp_source: torch.Tensor, kp_driving: torch.Tensor) -> torch.Tensor:
        """ conduct the stitching
        kp_source: Bxnum_kpx3
        kp_driving: Bxnum_kpx3
        """

        if self.stitching_retargeting_module is not None:

            bs, num_kp = kp_source.shape[:2]

            kp_driving_new = kp_driving.clone()
            delta = self.stitch(kp_source, kp_driving_new)

            delta_exp = delta[..., :3*num_kp].reshape(bs, num_kp, 3)  # 1x20x3
            delta_tx_ty = delta[..., 3*num_kp:3*num_kp+2].reshape(bs, 1, 2)  # 1x1x2

            kp_driving_new += delta_exp
            kp_driving_new[..., :2] += delta_tx_ty

            return kp_driving_new

        return kp_driving

    def warp_decode(self, feature_3d: torch.Tensor, kp_source: torch.Tensor, kp_driving: torch.Tensor) -> torch.Tensor:
        """ get the image after the warping of the implicit keypoints
        feature_3d: Bx32x16x64x64, feature volume
        kp_source: BxNx3
        kp_driving: BxNx3
        """
        # The line 18 in Algorithm 1: D(W(f_s; x_s, x′_d,i)）
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.cfg.flag_use_half_precision):
                # get decoder input
                ret_dct = self.warping_module(feature_3d, kp_source=kp_source, kp_driving=kp_driving)
                # decode
                ret_dct['out'] = self.spade_generator(feature=ret_dct['out'])

            # float the dict
            if self.cfg.flag_use_half_precision:
                for k, v in ret_dct.items():
                    if isinstance(v, torch.Tensor):
                        ret_dct[k] = v.float()

        return ret_dct

    def parse_output(self, out: torch.Tensor) -> np.ndarray:
        """ construct the output as standard
        return: 1xHxWx3, uint8
        """
        out = np.transpose(out.data.cpu().numpy(), [0, 2, 3, 1])  # 1x3xHxW -> 1xHxWx3
        out = np.clip(out, 0, 1)  # clip to 0~1
        out = np.clip(out * 255, 0, 255).astype(np.uint8)  # 0~1 -> 0~255

        return out

    def calc_retargeting_ratio(self, source_lmk, driving_lmk_lst):
        input_eye_ratio_lst = []
        input_lip_ratio_lst = []
        for lmk in driving_lmk_lst:
            # for eyes retargeting
            input_eye_ratio_lst.append(calc_eye_close_ratio(lmk[None]))
            # for lip retargeting
            input_lip_ratio_lst.append(calc_lip_close_ratio(lmk[None]))
        return input_eye_ratio_lst, input_lip_ratio_lst

    def calc_combined_eye_ratio(self, input_eye_ratio, source_lmk):
        eye_close_ratio = calc_eye_close_ratio(source_lmk[None])
        eye_close_ratio_tensor = torch.from_numpy(eye_close_ratio).float().cuda(self.device_id)
        input_eye_ratio_tensor = torch.Tensor([input_eye_ratio[0][0]]).reshape(1, 1).cuda(self.device_id)
        # [c_s,eyes, c_d,eyes,i]
        combined_eye_ratio_tensor = torch.cat([eye_close_ratio_tensor, input_eye_ratio_tensor], dim=1)
        return combined_eye_ratio_tensor

    def calc_combined_lip_ratio(self, input_lip_ratio, source_lmk):
        lip_close_ratio = calc_lip_close_ratio(source_lmk[None])
        lip_close_ratio_tensor = torch.from_numpy(lip_close_ratio).float().cuda(self.device_id)
        # [c_s,lip, c_d,lip,i]
        input_lip_ratio_tensor = torch.Tensor([input_lip_ratio[0]]).cuda(self.device_id)
        if input_lip_ratio_tensor.shape != [1, 1]:
            input_lip_ratio_tensor = input_lip_ratio_tensor.reshape(1, 1)
        combined_lip_ratio_tensor = torch.cat([lip_close_ratio_tensor, input_lip_ratio_tensor], dim=1)
        return combined_lip_ratio_tensor
