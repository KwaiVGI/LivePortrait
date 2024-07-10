# coding: utf-8

import numpy as np
import os.path as osp
from typing import List, Union, Tuple
from dataclasses import dataclass, field
import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)

from ..config.crop_config import CropConfig
from .landmark_runner import LandmarkRunner
from .face_analysis_diy import FaceAnalysisDIY
from .crop import crop_image, crop_image_by_bbox, parse_bbox_from_landmark, average_bbox_lst
from .rprint import rlog as log
from .io import contiguous


def make_abs_path(fn):
    return osp.join(osp.dirname(osp.realpath(__file__)), fn)


@dataclass
class Trajectory:
    start: int = -1  # start frame
    end: int = -1  # end frame
    lmk_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # lmk list
    bbox_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # bbox list

    frame_rgb_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # frame list

    lmk_crop_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # lmk list
    frame_rgb_crop_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # frame crop list


class Cropper(object):
    def __init__(self, **kwargs) -> None:
        device_id = kwargs.get('device_id', 0)
        flag_force_cpu = kwargs.get('flag_force_cpu', False)
        if flag_force_cpu:
            device = 'cpu'
            face_analysis_wrapper_provicer = ['CPUExecutionProvider']
        else:
            device = 'cuda'
            face_analysis_wrapper_provicer = ["CUDAExecutionProvider"]
        self.landmark_runner = LandmarkRunner(
            ckpt_path=make_abs_path('../../pretrained_weights/liveportrait/landmark.onnx'),
            onnx_provider=device,
            device_id=device_id
        )
        self.landmark_runner.warmup()


        self.face_analysis_wrapper = FaceAnalysisDIY(
            name='buffalo_l',
            root=make_abs_path('../../pretrained_weights/insightface'),
            providers=face_analysis_wrapper_provicer
        )
        self.face_analysis_wrapper.prepare(ctx_id=device_id, det_size=(512, 512))
        self.face_analysis_wrapper.warmup()

        self.crop_cfg: CropConfig = kwargs.get('crop_cfg', None)

    def update_config(self, user_args):
        for k, v in user_args.items():
            if hasattr(self.crop_cfg, k):
                setattr(self.crop_cfg, k, v)

    def crop_source_image(self, img_rgb_: np.ndarray, crop_cfg: CropConfig):
        # crop a source image and get neccessary information
        img_rgb = img_rgb_.copy()  # copy it

        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        src_face = self.face_analysis_wrapper.get(
            img_bgr,
            flag_do_landmark_2d_106=True,
            direction=crop_cfg.direction,
            max_face_num=crop_cfg.max_face_num,
        )

        if len(src_face) == 0:
            log('No face detected in the source image.')
            return None
        elif len(src_face) > 1:
            log(f'More than one face detected in the image, only pick one face by rule {crop_cfg.direction}.')

        # NOTE: temporarily only pick the first face, to support multiple face in the future
        src_face = src_face[0]
        lmk = src_face.landmark_2d_106  # this is the 106 landmarks from insightface

        # crop the face
        ret_dct = crop_image(
            img_rgb,  # ndarray
            lmk,  # 106x2 or Nx2
            dsize=crop_cfg.dsize,
            scale=crop_cfg.scale,
            vx_ratio=crop_cfg.vx_ratio,
            vy_ratio=crop_cfg.vy_ratio,
        )

        lmk = self.landmark_runner.run(img_rgb, lmk)
        ret_dct['lmk_crop'] = lmk

        # update a 256x256 version for network input
        ret_dct['img_crop_256x256'] = cv2.resize(ret_dct['img_crop'], (256, 256), interpolation=cv2.INTER_AREA)
        ret_dct['lmk_crop_256x256'] = ret_dct['lmk_crop'] * 256 / crop_cfg.dsize

        return ret_dct

    def crop_driving_video(self, driving_rgb_lst, **kwargs):
        """Tracking based landmarks/alignment and cropping"""
        trajectory = Trajectory()
        direction = kwargs.get('direction', 'large-small')
        for idx, frame_rgb in enumerate(driving_rgb_lst):
            if idx == 0 or trajectory.start == -1:
                src_face = self.face_analysis_wrapper.get(
                    contiguous(frame_rgb[..., ::-1]),
                    flag_do_landmark_2d_106=True,
                    direction=direction
                )
                if len(src_face) == 0:
                    log(f'No face detected in the frame #{idx}')
                    continue
                elif len(src_face) > 1:
                    log(f'More than one face detected in the driving frame_{idx}, only pick one face by rule {direction}.')
                src_face = src_face[0]
                lmk = src_face.landmark_2d_106
                lmk = self.landmark_runner.run(frame_rgb, lmk)
                trajectory.start, trajectory.end = idx, idx
            else:
                lmk = self.landmark_runner.run(frame_rgb, trajectory.lmk_lst[-1])
                trajectory.end = idx

            trajectory.lmk_lst.append(lmk)
            ret_bbox = parse_bbox_from_landmark(lmk, scale=self.crop_cfg.scale_crop_video, vx_ratio_crop_video=self.crop_cfg.vx_ratio_crop_video, vy_ratio=self.crop_cfg.vy_ratio_crop_video)['bbox']
            bbox = [ret_bbox[0, 0], ret_bbox[0, 1], ret_bbox[2, 0], ret_bbox[2, 1]]  # 4,
            trajectory.bbox_lst.append(bbox)  # bbox
            trajectory.frame_rgb_lst.append(frame_rgb)

        global_bbox = average_bbox_lst(trajectory.bbox_lst)

        for idx, (frame_rgb, lmk) in enumerate(zip(trajectory.frame_rgb_lst, trajectory.lmk_lst)):
            ret_dct = crop_image_by_bbox(
                frame_rgb,
                global_bbox,
                lmk=lmk,
                dsize=kwargs.get('dsize', 512),
                flag_rot=False,
                borderValue=(0, 0, 0),
            )
            trajectory.frame_rgb_crop_lst.append(ret_dct['img_crop'])
            trajectory.lmk_crop_lst.append(ret_dct['lmk_crop'])

        return {
            'frame_crop_lst': trajectory.frame_rgb_crop_lst,
            'lmk_crop_lst': trajectory.lmk_crop_lst,
        }

    def calc_lmks_from_cropped_video(self, driving_rgb_crop_lst, **kwargs):
        """Tracking based landmarks/alignment"""
        trajectory = Trajectory()
        direction = kwargs.get('direction', 'large-small')

        for idx, frame_rgb_crop in enumerate(driving_rgb_crop_lst):
            if idx == 0 or trajectory.start == -1:
                src_face = self.face_analysis_wrapper.get(
                    contiguous(frame_rgb_crop[..., ::-1]),  # convert to BGR
                    flag_do_landmark_2d_106=True,
                    direction=direction
                )
                if len(src_face) == 0:
                    log(f'No face detected in the frame #{idx}')
                    raise Exception(f'No face detected in the frame #{idx}')
                elif len(src_face) > 1:
                    log(f'More than one face detected in the driving frame_{idx}, only pick one face by rule {direction}.')
                src_face = src_face[0]
                lmk = src_face.landmark_2d_106
                lmk = self.landmark_runner.run(frame_rgb_crop, lmk)
                trajectory.start, trajectory.end = idx, idx
            else:
                lmk = self.landmark_runner.run(frame_rgb_crop, trajectory.lmk_lst[-1])
                trajectory.end = idx

            trajectory.lmk_lst.append(lmk)
        return trajectory.lmk_lst
