# coding: utf-8

import os.path as osp
import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)
import torch
import numpy as np
import onnxruntime
from .timer import Timer
from .rprint import rlog
from .crop import crop_image, _transform_pts


def make_abs_path(fn):
    return osp.join(osp.dirname(osp.realpath(__file__)), fn)


def to_ndarray(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy()
    elif isinstance(obj, np.ndarray):
        return obj
    else:
        return np.array(obj)


class LandmarkRunner(object):
    """landmark runner"""

    def __init__(self, **kwargs):
        ckpt_path = kwargs.get('ckpt_path')
        onnx_provider = kwargs.get('onnx_provider', 'cuda')  # 默认用cuda
        device_id = kwargs.get('device_id', 0)
        self.dsize = kwargs.get('dsize', 224)
        self.timer = Timer()

        if onnx_provider.lower() == 'cuda':
            self.session = onnxruntime.InferenceSession(
                ckpt_path, providers=[
                    ('CUDAExecutionProvider', {'device_id': device_id})
                ]
            )
        elif onnx_provider.lower() == 'mps':
            self.session = onnxruntime.InferenceSession(
                ckpt_path, providers=[
                    'CoreMLExecutionProvider'
                ]
            )
        else:
            opts = onnxruntime.SessionOptions()
            opts.intra_op_num_threads = 4  # 默认线程数为 4
            self.session = onnxruntime.InferenceSession(
                ckpt_path, providers=['CPUExecutionProvider'],
                sess_options=opts
            )

    def _run(self, inp):
        out = self.session.run(None, {'input': inp})
        return out

    def run(self, img_rgb: np.ndarray, lmk=None):
        if lmk is not None:
            crop_dct = crop_image(img_rgb, lmk, dsize=self.dsize, scale=1.5, vy_ratio=-0.1)
            img_crop_rgb = crop_dct['img_crop']
        else:
            # NOTE: force resize to 224x224, NOT RECOMMEND!
            img_crop_rgb = cv2.resize(img_rgb, (self.dsize, self.dsize))
            scale = max(img_rgb.shape[:2]) / self.dsize
            crop_dct = {
                'M_c2o': np.array([
                    [scale, 0., 0.],
                    [0., scale, 0.],
                    [0., 0., 1.],
                ], dtype=np.float32),
            }

        inp = (img_crop_rgb.astype(np.float32) / 255.).transpose(2, 0, 1)[None, ...]  # HxWx3 (BGR) -> 1x3xHxW (RGB!)

        out_lst = self._run(inp)
        out_pts = out_lst[2]

        # 2d landmarks 203 points
        lmk = to_ndarray(out_pts[0]).reshape(-1, 2) * self.dsize  # scale to 0-224
        lmk = _transform_pts(lmk, M=crop_dct['M_c2o'])

        return lmk

    def warmup(self):
        self.timer.tic()

        dummy_image = np.zeros((1, 3, self.dsize, self.dsize), dtype=np.float32)

        _ = self._run(dummy_image)

        elapse = self.timer.toc()
        rlog(f'LandmarkRunner warmup time: {elapse:.3f}s')
