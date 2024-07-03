# coding: utf-8

import numpy as np
import os.path as osp
from typing import List, Union, Tuple
from dataclasses import dataclass, field
import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)

from .landmark_runner import LandmarkRunner
from .face_analysis_diy import FaceAnalysisDIY
from .helper import prefix
from .crop import crop_image, crop_image_by_bbox, parse_bbox_from_landmark, average_bbox_lst
from .timer import Timer
from .rprint import rlog as log
from .io import load_image_rgb
from .video import VideoWriter, get_fps, change_video_fps


def make_abs_path(fn):
    return osp.join(osp.dirname(osp.realpath(__file__)), fn)


@dataclass
class Trajectory:
    start: int = -1  # 起始帧 闭区间
    end: int = -1  # 结束帧 闭区间
    lmk_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # lmk list
    bbox_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # bbox list
    frame_rgb_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # frame list
    frame_rgb_crop_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # frame crop list


class Cropper(object):
    def __init__(self, **kwargs) -> None:
        device_id = kwargs.get('device_id', 0)
        self.landmark_runner = LandmarkRunner(
            ckpt_path=make_abs_path('../../pretrained_weights/liveportrait/landmark.onnx'),
            onnx_provider='cuda',
            device_id=device_id
        )
        self.landmark_runner.warmup()

        self.face_analysis_wrapper = FaceAnalysisDIY(
            name='buffalo_l',
            root=make_abs_path('../../pretrained_weights/insightface'),
            providers=["CUDAExecutionProvider"]
        )
        self.face_analysis_wrapper.prepare(ctx_id=device_id, det_size=(512, 512))
        self.face_analysis_wrapper.warmup()

        self.crop_cfg = kwargs.get('crop_cfg', None)

    def update_config(self, user_args):
        for k, v in user_args.items():
            if hasattr(self.crop_cfg, k):
                setattr(self.crop_cfg, k, v)

    def crop_single_image(self, obj, **kwargs):
        direction = kwargs.get('direction', 'large-small')

        # crop and align a single image
        if isinstance(obj, str):
            img_rgb = load_image_rgb(obj)
        elif isinstance(obj, np.ndarray):
            img_rgb = obj

        src_face = self.face_analysis_wrapper.get(
            img_rgb,
            flag_do_landmark_2d_106=True,
            direction=direction
        )

        if len(src_face) == 0:
            log('No face detected in the source image.')
            raise Exception("No face detected in the source image!")
        elif len(src_face) > 1:
            log(f'More than one face detected in the image, only pick one face by rule {direction}.')

        src_face = src_face[0]
        pts = src_face.landmark_2d_106

        # crop the face
        ret_dct = crop_image(
            img_rgb,  # ndarray
            pts,  # 106x2 or Nx2
            dsize=kwargs.get('dsize', 512),
            scale=kwargs.get('scale', 2.3),
            vy_ratio=kwargs.get('vy_ratio', -0.15),
        )
        # update a 256x256 version for network input or else
        ret_dct['img_crop_256x256'] = cv2.resize(ret_dct['img_crop'], (256, 256), interpolation=cv2.INTER_AREA)
        ret_dct['pt_crop_256x256'] = ret_dct['pt_crop'] * 256 / kwargs.get('dsize', 512)

        recon_ret = self.landmark_runner.run(img_rgb, pts)
        lmk = recon_ret['pts']
        ret_dct['lmk_crop'] = lmk

        return ret_dct

    def get_retargeting_lmk_info(self, driving_rgb_lst):
        # TODO: implement a tracking-based version
        driving_lmk_lst = []
        for driving_image in driving_rgb_lst:
            ret_dct = self.crop_single_image(driving_image)
            driving_lmk_lst.append(ret_dct['lmk_crop'])
        return driving_lmk_lst

    def make_video_clip(self, driving_rgb_lst, output_path, output_fps=30, **kwargs):
        trajectory = Trajectory()
        direction = kwargs.get('direction', 'large-small')
        for idx, driving_image in enumerate(driving_rgb_lst):
            if idx == 0 or trajectory.start == -1:
                src_face = self.face_analysis_wrapper.get(
                    driving_image,
                    flag_do_landmark_2d_106=True,
                    direction=direction
                )
                if len(src_face) == 0:
                    # No face detected in the driving_image
                    continue
                elif len(src_face) > 1:
                    log(f'More than one face detected in the driving frame_{idx}, only pick one face by rule {direction}.')
                src_face = src_face[0]
                pts = src_face.landmark_2d_106
                lmk_203 = self.landmark_runner(driving_image, pts)['pts']
                trajectory.start, trajectory.end = idx, idx
            else:
                lmk_203 = self.face_recon_wrapper(driving_image, trajectory.lmk_lst[-1])['pts']
                trajectory.end = idx

            trajectory.lmk_lst.append(lmk_203)
            ret_bbox = parse_bbox_from_landmark(lmk_203, scale=self.crop_cfg.globalscale, vy_ratio=elf.crop_cfg.vy_ratio)['bbox']
            bbox = [ret_bbox[0, 0], ret_bbox[0, 1], ret_bbox[2, 0], ret_bbox[2, 1]]  # 4,
            trajectory.bbox_lst.append(bbox)  # bbox
            trajectory.frame_rgb_lst.append(driving_image)

        global_bbox = average_bbox_lst(trajectory.bbox_lst)
        for idx, (frame_rgb, lmk) in enumerate(zip(trajectory.frame_rgb_lst, trajectory.lmk_lst)):
            ret_dct = crop_image_by_bbox(
                frame_rgb, global_bbox, lmk=lmk,
                dsize=self.video_crop_cfg.dsize, flag_rot=self.video_crop_cfg.flag_rot, borderValue=self.video_crop_cfg.borderValue
            )
            frame_rgb_crop = ret_dct['img_crop']
