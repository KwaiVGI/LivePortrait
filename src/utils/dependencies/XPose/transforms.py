# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
"""
import os
import sys
import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from util.box_ops import box_xyxy_to_cxcywh
from util.misc import interpolate


def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    if target is not None:
        target = target.copy()
        i, j, h, w = region
        id2catname = target["id2catname"]
        caption_list = target["caption_list"]
        target["size"] = torch.tensor([h, w])

        fields = ["labels", "area", "iscrowd", "positive_map","keypoints"]

        if "boxes" in target:
            boxes = target["boxes"]
            max_size = torch.as_tensor([w, h], dtype=torch.float32)
            cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
            cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
            cropped_boxes = cropped_boxes.clamp(min=0)
            area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
            target["boxes"] = cropped_boxes.reshape(-1, 4)
            target["area"] = area
            fields.append("boxes")

        if "masks" in target:
            # FIXME should we update the area here if there are no boxes?
            target['masks'] = target['masks'][:, i:i + h, j:j + w]
            fields.append("masks")


        # remove elements for which the boxes or masks that have zero area
        if "boxes" in target or "masks" in target:
            # favor boxes selection when defining which elements to keep
            # this is compatible with previous implementation
            if "boxes" in target:
                cropped_boxes = target['boxes'].reshape(-1, 2, 2)
                keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
            else:
                keep = target['masks'].flatten(1).any(1)

            for field in fields:
                if field in target:
                    target[field] = target[field][keep]

        if os.environ.get('IPDB_SHILONG_DEBUG', None) == 'INFO':
            # for debug and visualization only.
            if 'strings_positive' in target:
                target['strings_positive'] = [_i for _i, _j in zip(target['strings_positive'], keep) if _j]


        if "keypoints" in target:
            max_size = torch.as_tensor([w, h], dtype=torch.float32)
            keypoints = target["keypoints"]
            cropped_keypoints = keypoints.view(-1, 3)[:,:2] - torch.as_tensor([j, i])
            cropped_keypoints = torch.min(cropped_keypoints, max_size)
            cropped_keypoints = cropped_keypoints.clamp(min=0)
            cropped_keypoints = torch.cat([cropped_keypoints, keypoints.view(-1, 3)[:,2].unsqueeze(1)], dim=1)
            target["keypoints"] = cropped_keypoints.view(target["keypoints"].shape[0], target["keypoints"].shape[1], 3)

        target["id2catname"] = id2catname
        target["caption_list"] = caption_list

    return cropped_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    if target is not None:
        target = target.copy()
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
            target["boxes"] = boxes

        if "masks" in target:
            target['masks'] = target['masks'].flip(-1)


        if "keypoints" in target:
            dataset_name=target["dataset_name"]
            if dataset_name == "coco_person" or dataset_name == "macaque":
                flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
                                   [9, 10], [11, 12], [13, 14], [15, 16]]

            elif dataset_name=="animalkindom_ak_P1_animal":
                flip_pairs = [[1, 2], [4, 5],[7,8],[9,10],[11,12],[14,15],[16,17],[18,19]]

            elif dataset_name=="animalweb_animal":
                flip_pairs = [[0, 3], [1, 2], [5, 6]]

            elif dataset_name=="face":
                flip_pairs = [
                                [0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11], [6, 10], [7, 9],
                                [17, 26], [18, 25], [19, 24], [20, 23], [21, 22],
                                [31, 35], [32, 34],
                                [36, 45], [37, 44], [38, 43], [39, 42], [40, 47], [41, 46],
                                [48, 54], [49, 53], [50, 52],
                                [55, 59], [56, 58],
                                [60, 64], [61, 63],
                                [65, 67]
                            ]

            elif dataset_name=="hand":
                flip_pairs = []

            elif dataset_name=="foot":
                flip_pairs = []

            elif dataset_name=="locust":
                flip_pairs = [[5, 20], [6, 21], [7, 22], [8, 23], [9, 24], [10, 25], [11, 26], [12, 27], [13, 28], [14, 29], [15, 30], [16, 31], [17, 32], [18, 33], [19, 34]]

            elif dataset_name=="fly":
                flip_pairs = [[1, 2], [6, 18], [7, 19], [8, 20], [9, 21], [10, 22], [11, 23], [12, 24], [13, 25], [14, 26], [15, 27], [16, 28], [17, 29], [30, 31]]

            elif dataset_name == "ap_36k_animal" or dataset_name == "ap_10k_animal":
                flip_pairs = [[0, 1],[5, 8], [6, 9], [7, 10], [11, 14], [12, 15], [13, 16]]



            keypoints = target["keypoints"]
            keypoints[:,:,0] = w - keypoints[:,:, 0]-1
            for pair in flip_pairs:
                keypoints[:,pair[0], :], keypoints[:,pair[1], :] = keypoints[:,pair[1], :], keypoints[:,pair[0], :].clone()
            target["keypoints"] = keypoints
    return flipped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area


    if "keypoints" in target:
        keypoints = target["keypoints"]
        scaled_keypoints = keypoints * torch.as_tensor([ratio_width, ratio_height, 1])
        target["keypoints"] = scaled_keypoints

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image.size[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target


class ResizeDebug(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        return resize(img, target, self.size)


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int, respect_boxes: bool = False):
        # respect_boxes:    True to keep all boxes
        #                   False to tolerence box filter
        self.min_size = min_size
        self.max_size = max_size
        self.respect_boxes = respect_boxes

    def __call__(self, img: PIL.Image.Image, target: dict):
        init_boxes = len(target["boxes"]) if (target is not None and "boxes" in target) else 0
        max_patience = 10
        for i in range(max_patience):
            w = random.randint(self.min_size, min(img.width, self.max_size))
            h = random.randint(self.min_size, min(img.height, self.max_size))
            region = T.RandomCrop.get_params(img, [h, w])
            result_img, result_target = crop(img, target, region)
            if target is not None:
                if not self.respect_boxes or len(result_target["boxes"]) == init_boxes or i == max_patience - 1:
                    return result_img, result_target
        return result_img, result_target


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes

        if "area" in target:
            area = target["area"]
            area = area / (torch.tensor(w, dtype=torch.float32)*torch.tensor(h, dtype=torch.float32))
            target["area"] = area

        if "keypoints" in target:
            keypoints = target["keypoints"]
            V = keypoints[:, :, 2]
            V[V == 2] = 1
            Z=keypoints[:, :, :2]
            Z = Z.contiguous().view(-1, 2 * V.shape[-1])
            Z = Z / torch.tensor([w, h] * V.shape[-1], dtype=torch.float32)
            target["valid_kpt_num"] = V.shape[1]
            Z_pad = torch.zeros(Z.shape[0],68 * 2 - Z.shape[1])
            V_pad = torch.zeros(V.shape[0],68 - V.shape[1])
            V=torch.cat([V, V_pad], dim=1)
            Z=torch.cat([Z, Z_pad], dim=1)
            all_keypoints = torch.cat([Z, V], dim=1)
            target["keypoints"] = all_keypoints


        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
