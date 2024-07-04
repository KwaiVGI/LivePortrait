# coding: utf-8

import os
from glob import glob
import os.path as osp
import imageio
import numpy as np
import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)


def load_image_rgb(image_path: str):
    if not osp.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_driving_info(driving_info):
    driving_video_ori = []

    def load_images_from_directory(directory):
        image_paths = sorted(glob(osp.join(directory, '*.png')) + glob(osp.join(directory, '*.jpg')))
        return [load_image_rgb(im_path) for im_path in image_paths]

    def load_images_from_video(file_path):
        reader = imageio.get_reader(file_path)
        return [image for idx, image in enumerate(reader)]

    if osp.isdir(driving_info):
        driving_video_ori = load_images_from_directory(driving_info)
    elif osp.isfile(driving_info):
        driving_video_ori = load_images_from_video(driving_info)

    return driving_video_ori


def contiguous(obj):
    if not obj.flags.c_contiguous:
        obj = obj.copy(order="C")
    return obj


def resize_to_limit(img: np.ndarray, max_dim=1920, n=2):
    """
    ajust the size of the image so that the maximum dimension does not exceed max_dim, and the width and the height of the image are multiples of n.
    :param img: the image to be processed.
    :param max_dim: the maximum dimension constraint.
    :param n: the number that needs to be multiples of.
    :return: the adjusted image.
    """
    h, w = img.shape[:2]

    # ajust the size of the image according to the maximum dimension
    if max_dim > 0 and max(h, w) > max_dim:
        if h > w:
            new_h = max_dim
            new_w = int(w * (max_dim / h))
        else:
            new_w = max_dim
            new_h = int(h * (max_dim / w))
        img = cv2.resize(img, (new_w, new_h))

    # ensure that the image dimensions are multiples of n
    n = max(n, 1)
    new_h = img.shape[0] - (img.shape[0] % n)
    new_w = img.shape[1] - (img.shape[1] % n)

    if new_h == 0 or new_w == 0:
        # when the width or height is less than n, no need to process
        return img

    if new_h != img.shape[0] or new_w != img.shape[1]:
        img = img[:new_h, :new_w]

    return img


def load_img_online(obj, mode="bgr", **kwargs):
    max_dim = kwargs.get("max_dim", 1920)
    n = kwargs.get("n", 2)
    if isinstance(obj, str):
        if mode.lower() == "gray":
            img = cv2.imread(obj, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(obj, cv2.IMREAD_COLOR)
    else:
        img = obj

    # Resize image to satisfy constraints
    img = resize_to_limit(img, max_dim=max_dim, n=n)

    if mode.lower() == "bgr":
        return contiguous(img)
    elif mode.lower() == "rgb":
        return contiguous(img[..., ::-1])
    else:
        raise Exception(f"Unknown mode {mode}")
