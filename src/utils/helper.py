# coding: utf-8

"""
utility functions and classes to handle feature extraction and model loading
"""

import os
import os.path as osp
import torch
from collections import OrderedDict
import numpy as np
import cv2

from ..modules.spade_generator import SPADEDecoder
from ..modules.warping_network import WarpingNetwork
from ..modules.motion_extractor import MotionExtractor
from ..modules.appearance_feature_extractor import AppearanceFeatureExtractor
from ..modules.stitching_retargeting_network import StitchingRetargetingNetwork


def suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind(".")
    if pos == -1:
        return ""
    return filename[pos + 1:]


def prefix(filename):
    """a.jpg -> a"""
    pos = filename.rfind(".")
    if pos == -1:
        return filename
    return filename[:pos]


def basename(filename):
    """a/b/c.jpg -> c"""
    return prefix(osp.basename(filename))


def remove_suffix(filepath):
    """a/b/c.jpg -> a/b/c"""
    return osp.join(osp.dirname(filepath), basename(filepath))


def is_image(file_path):
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
    return file_path.lower().endswith(image_extensions)


def is_video(file_path):
    if file_path.lower().endswith((".mp4", ".mov", ".avi", ".webm")) or osp.isdir(file_path):
        return True
    return False


def is_template(file_path):
    if file_path.endswith(".pkl"):
        return True
    return False


def mkdir(d, log=False):
    # return self-assined `d`, for one line code
    if not osp.exists(d):
        os.makedirs(d, exist_ok=True)
        if log:
            print(f"Make dir: {d}")
    return d


def squeeze_tensor_to_numpy(tensor):
    out = tensor.data.squeeze(0).cpu().numpy()
    return out


def dct2device(dct: dict, device):
    for key in dct:
        dct[key] = torch.tensor(dct[key]).to(device)
    return dct


def concat_feat(kp_source: torch.Tensor, kp_driving: torch.Tensor) -> torch.Tensor:
    """
    kp_source: (bs, k, 3)
    kp_driving: (bs, k, 3)
    Return: (bs, 2k*3)
    """
    bs_src = kp_source.shape[0]
    bs_dri = kp_driving.shape[0]
    assert bs_src == bs_dri, 'batch size must be equal'

    feat = torch.cat([kp_source.view(bs_src, -1), kp_driving.view(bs_dri, -1)], dim=1)
    return feat


def remove_ddp_dumplicate_key(state_dict):
    state_dict_new = OrderedDict()
    for key in state_dict.keys():
        state_dict_new[key.replace('module.', '')] = state_dict[key]
    return state_dict_new


def load_model(ckpt_path, model_config, device, model_type):
    model_params = model_config['model_params'][f'{model_type}_params']

    if model_type == 'appearance_feature_extractor':
        model = AppearanceFeatureExtractor(**model_params).to(device)
    elif model_type == 'motion_extractor':
        model = MotionExtractor(**model_params).to(device)
    elif model_type == 'warping_module':
        model = WarpingNetwork(**model_params).to(device)
    elif model_type == 'spade_generator':
        model = SPADEDecoder(**model_params).to(device)
    elif model_type == 'stitching_retargeting_module':
        # Special handling for stitching and retargeting module
        config = model_config['model_params']['stitching_retargeting_module_params']
        checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)

        stitcher = StitchingRetargetingNetwork(**config.get('stitching'))
        stitcher.load_state_dict(remove_ddp_dumplicate_key(checkpoint['retarget_shoulder']))
        stitcher = stitcher.to(device)
        stitcher.eval()

        retargetor_lip = StitchingRetargetingNetwork(**config.get('lip'))
        retargetor_lip.load_state_dict(remove_ddp_dumplicate_key(checkpoint['retarget_mouth']))
        retargetor_lip = retargetor_lip.to(device)
        retargetor_lip.eval()

        retargetor_eye = StitchingRetargetingNetwork(**config.get('eye'))
        retargetor_eye.load_state_dict(remove_ddp_dumplicate_key(checkpoint['retarget_eye']))
        retargetor_eye = retargetor_eye.to(device)
        retargetor_eye.eval()

        return {
            'stitching': stitcher,
            'lip': retargetor_lip,
            'eye': retargetor_eye
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(torch.load(ckpt_path, map_location=lambda storage, loc: storage))
    model.eval()
    return model


def load_description(fp):
    with open(fp, 'r', encoding='utf-8') as f:
        content = f.read()
    return content


def is_square_video(video_path):
    video = cv2.VideoCapture(video_path)

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video.release()
    # if width != height:
        # gr.Info(f"Uploaded video is not square, force do crop (driving) to be True")

    return width == height
