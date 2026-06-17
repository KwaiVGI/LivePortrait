# coding: utf-8

import os.path as osp
import torch
import numpy as np
import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)

from PIL import Image
from typing import List, Tuple, Union
from dataclasses import dataclass, field

from ..config.crop_config import CropConfig
from .crop import (
    average_bbox_lst,
    crop_image,
    crop_image_by_bbox,
    parse_bbox_from_landmark,
)
from .io import contiguous
from .rprint import rlog as log
from .face_analysis_diy import FaceAnalysisDIY
from .human_landmark_runner import LandmarkRunner as HumanLandmark

def make_abs_path(fn):
    return osp.join(osp.dirname(osp.realpath(__file__)), fn)


@dataclass
class Trajectory:
    start: int = -1  # start frame
    end: int = -1  # end frame
    lmk_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # lmk list
    bbox_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # bbox list
    M_c2o_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # M_c2o list

    frame_rgb_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # frame list
    lmk_crop_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # lmk list
    frame_rgb_crop_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # frame crop list


class Cropper(object):
    def __init__(self, **kwargs) -> None:
        self.crop_cfg: CropConfig = kwargs.get("crop_cfg", None)
        self.image_type = kwargs.get("image_type", 'human_face')
        device_id = kwargs.get("device_id", 0)
        flag_force_cpu = kwargs.get("flag_force_cpu", False)
        if flag_force_cpu:
            device = "cpu"
            face_analysis_wrapper_provider = ["CPUExecutionProvider"]
        else:
            try:
                if torch.backends.mps.is_available():
                    # Shape inference currently fails with CoreMLExecutionProvider
                    # for the retinaface model
                    device = "mps"
                    face_analysis_wrapper_provider = ["CPUExecutionProvider"]
                else:
                    device = "cuda"
                    face_analysis_wrapper_provider = ["CUDAExecutionProvider"]
            except:
                    device = "cuda"
                    face_analysis_wrapper_provider = ["CUDAExecutionProvider"]
        self.face_analysis_wrapper = FaceAnalysisDIY(
                    name="buffalo_l",
                    root=self.crop_cfg.insightface_root,
                    providers=face_analysis_wrapper_provider,
                )
        self.face_analysis_wrapper.prepare(ctx_id=device_id, det_size=(512, 512), det_thresh=self.crop_cfg.det_thresh)
        self.face_analysis_wrapper.warmup()

        self.human_landmark_runner = HumanLandmark(
            ckpt_path=self.crop_cfg.landmark_ckpt_path,
            onnx_provider=device,
            device_id=device_id,
        )
        self.human_landmark_runner.warmup()

        if self.image_type == "animal_face":
            from .animal_landmark_runner import XPoseRunner as AnimalLandmarkRunner
            self.animal_landmark_runner = AnimalLandmarkRunner(
                    model_config_path=self.crop_cfg.xpose_config_file_path,
                    model_checkpoint_path=self.crop_cfg.xpose_ckpt_path,
                    embeddings_cache_path=self.crop_cfg.xpose_embedding_cache_path,
                    flag_use_half_precision=kwargs.get("flag_use_half_precision", True),
                )
            self.animal_landmark_runner.warmup()

    def update_config(self, user_args):
        for k, v in user_args.items():
            if hasattr(self.crop_cfg, k):
                setattr(self.crop_cfg, k, v)

    def crop_source_image(self, img_rgb_: np.ndarray, crop_cfg: CropConfig):
        # crop a source image and get neccessary information
        img_rgb = img_rgb_.copy()  # copy it
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        if self.image_type == "human_face":
            src_face = self.face_analysis_wrapper.get(
                img_bgr,
                flag_do_landmark_2d_106=True,
                direction=crop_cfg.direction,
                max_face_num=crop_cfg.max_face_num,
            )

            if len(src_face) == 0:
                log("No face detected in the source image.")
                return None
            elif len(src_face) > 1:
                log(f"More than one face detected in the image, only pick one face by rule {crop_cfg.direction}.")

            # NOTE: temporarily only pick the first face, to support multiple face in the future
            src_face = src_face[0]
            lmk = src_face.landmark_2d_106  # this is the 106 landmarks from insightface
        else:
            tmp_dct = {
                'animal_face_9': 'animal_face',
                'animal_face_68': 'face'
            }

            img_rgb_pil = Image.fromarray(img_rgb)
            lmk = self.animal_landmark_runner.run(
                img_rgb_pil,
                'face',
                tmp_dct[crop_cfg.animal_face_type],
                0,
                0
            )

        # crop the face
        ret_dct = crop_image(
            img_rgb,  # ndarray
            lmk,  # 106x2 or Nx2
            dsize=crop_cfg.dsize,
            scale=crop_cfg.scale,
            vx_ratio=crop_cfg.vx_ratio,
            vy_ratio=crop_cfg.vy_ratio,
            flag_do_rot=crop_cfg.flag_do_rot,
        )

        # update a 256x256 version for network input
        ret_dct["img_crop_256x256"] = cv2.resize(ret_dct["img_crop"], (256, 256), interpolation=cv2.INTER_AREA)
        if self.image_type == "human_face":
            lmk = self.human_landmark_runner.run(img_rgb, lmk)
            ret_dct["lmk_crop"] = lmk
            ret_dct["lmk_crop_256x256"] = ret_dct["lmk_crop"] * 256 / crop_cfg.dsize
        else:
            # 68x2 or 9x2
            ret_dct["lmk_crop"] = lmk

        return ret_dct

    def calc_lmk_from_cropped_image(self, img_rgb_, **kwargs):
        direction = kwargs.get("direction", "large-small")
        src_face = self.face_analysis_wrapper.get(
            contiguous(img_rgb_[..., ::-1]),  # convert to BGR
            flag_do_landmark_2d_106=True,
            direction=direction,
        )
        if len(src_face) == 0:
            log("No face detected in the source image.")
            return None
        elif len(src_face) > 1:
            log(f"More than one face detected in the image, only pick one face by rule {direction}.")
        src_face = src_face[0]
        lmk = src_face.landmark_2d_106
        lmk = self.human_landmark_runner.run(img_rgb_, lmk)

        return lmk

    # TODO: support skipping frame with NO FACE
    def crop_source_video(self, source_rgb_lst, crop_cfg: CropConfig, **kwargs):
        """Tracking based landmarks/alignment and cropping"""
        trajectory = Trajectory()
        direction = kwargs.get("direction", "large-small")
        for idx, frame_rgb in enumerate(source_rgb_lst):
            if idx == 0 or trajectory.start == -1:
                src_face = self.face_analysis_wrapper.get(
                    contiguous(frame_rgb[..., ::-1]),
                    flag_do_landmark_2d_106=True,
                    direction=crop_cfg.direction,
                    max_face_num=crop_cfg.max_face_num,
                )
                if len(src_face) == 0:
                    log(f"No face detected in the frame #{idx}")
                    continue
                elif len(src_face) > 1:
                    log(f"More than one face detected in the source frame_{idx}, only pick one face by rule {direction}.")
                src_face = src_face[0]
                lmk = src_face.landmark_2d_106
                lmk = self.human_landmark_runner.run(frame_rgb, lmk)
                trajectory.start, trajectory.end = idx, idx
            else:
                # TODO: add IOU check for tracking
                lmk = self.human_landmark_runner.run(frame_rgb, trajectory.lmk_lst[-1])
                trajectory.end = idx

            trajectory.lmk_lst.append(lmk)

            # crop the face
            ret_dct = crop_image(
                frame_rgb,  # ndarray
                lmk,  # 106x2 or Nx2
                dsize=crop_cfg.dsize,
                scale=crop_cfg.scale,
                vx_ratio=crop_cfg.vx_ratio,
                vy_ratio=crop_cfg.vy_ratio,
                flag_do_rot=crop_cfg.flag_do_rot,
            )

            # update a 256x256 version for network input
            ret_dct["img_crop_256x256"] = cv2.resize(ret_dct["img_crop"], (256, 256), interpolation=cv2.INTER_AREA)
            ret_dct["lmk_crop_256x256"] = ret_dct["pt_crop"] * 256 / crop_cfg.dsize

            trajectory.frame_rgb_crop_lst.append(ret_dct["img_crop_256x256"])
            trajectory.lmk_crop_lst.append(ret_dct["lmk_crop_256x256"])
            trajectory.M_c2o_lst.append(ret_dct['M_c2o'])

        return {
            "frame_crop_lst": trajectory.frame_rgb_crop_lst,
            "lmk_crop_lst": trajectory.lmk_crop_lst,
            "M_c2o_lst": trajectory.M_c2o_lst,
        }

    # Fast fix untrack bug
    def crop_driving_video(self, driving_rgb_lst, crop_cfg: CropConfig, **kwargs):
        """Tracking based landmarks/alignment and cropping"""
        trajectory = Trajectory()
        direction = kwargs.get("direction", "large-small")
        for idx, frame_rgb in enumerate(driving_rgb_lst):
            if idx == 0 or trajectory.start == -1:
                src_face = self.face_analysis_wrapper.get(
                    contiguous(frame_rgb[..., ::-1]),
                    flag_do_landmark_2d_106=True,
                    direction=crop_cfg.direction,
                    max_face_num=crop_cfg.max_face_num,
                )
                if len(src_face) == 0:
                    log(f"No face detected in the frame #{idx}")
                    continue
                elif len(src_face) > 1:
                    log(f"More than one face detected in the driving frame_{idx}, only pick one face by rule {direction}.")
                src_face = src_face[0]
                lmk = src_face.landmark_2d_106
                lmk = self.human_landmark_runner.run(frame_rgb, lmk)
                trajectory.start, trajectory.end = idx, idx
            else:
                lmk = self.human_landmark_runner.run(frame_rgb, trajectory.lmk_lst[-1])
                trajectory.end = idx

            trajectory.lmk_lst.append(lmk)

            # crop the face
            ret_dct = crop_image(
                frame_rgb,  # ndarray
                lmk,  # 106x2 or Nx2
                dsize=crop_cfg.dsize,
                scale=crop_cfg.scale_crop_driving_video,
                vx_ratio=crop_cfg.vx_ratio_crop_driving_video,
                vy_ratio=crop_cfg.vy_ratio_crop_driving_video,
                flag_do_rot=crop_cfg.flag_do_rot,
            )

            # update a 256x256 version for network input
            ret_dct["img_crop_256x256"] = cv2.resize(ret_dct["img_crop"], (256, 256), interpolation=cv2.INTER_AREA)
            ret_dct["lmk_crop_256x256"] = ret_dct["pt_crop"] * 256 / crop_cfg.dsize

            trajectory.frame_rgb_crop_lst.append(ret_dct["img_crop_256x256"])
            trajectory.lmk_crop_lst.append(ret_dct["lmk_crop_256x256"])
            # trajectory.M_c2o_lst.append(ret_dct['M_c2o'])

        return {
            "frame_crop_lst": trajectory.frame_rgb_crop_lst,
            "lmk_crop_lst": trajectory.lmk_crop_lst,
        }


    def calc_lmks_from_cropped_video(self, driving_rgb_crop_lst, **kwargs):
        """Tracking based landmarks/alignment"""
        trajectory = Trajectory()
        direction = kwargs.get("direction", "large-small")

        for idx, frame_rgb_crop in enumerate(driving_rgb_crop_lst):
            if idx == 0 or trajectory.start == -1:
                src_face = self.face_analysis_wrapper.get(
                    contiguous(frame_rgb_crop[..., ::-1]),  # convert to BGR
                    flag_do_landmark_2d_106=True,
                    direction=direction,
                )
                if len(src_face) == 0:
                    log(f"No face detected in the frame #{idx}")
                    raise Exception(f"No face detected in the frame #{idx}")
                elif len(src_face) > 1:
                    log(f"More than one face detected in the driving frame_{idx}, only pick one face by rule {direction}.")
                src_face = src_face[0]
                lmk = src_face.landmark_2d_106
                lmk = self.human_landmark_runner.run(frame_rgb_crop, lmk)
                trajectory.start, trajectory.end = idx, idx
            else:
                lmk = self.human_landmark_runner.run(frame_rgb_crop, trajectory.lmk_lst[-1])
                trajectory.end = idx

            trajectory.lmk_lst.append(lmk)
        return trajectory.lmk_lst
