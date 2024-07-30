# coding: utf-8

"""
face detectoin and alignment using XPose
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import io
import numpy as np
import torch
import clip
import src.utils.dependencies.XPose.transforms_uni as T
from src.utils.dependencies.XPose.models import build_model
from src.utils.dependencies.XPose.predefined_keypoints import *
from src.utils.dependencies.XPose.util import box_ops
from src.utils.dependencies.XPose.util.config import Config
from src.utils.dependencies.XPose.util.utils import clean_state_dict
from torchvision.ops import nms
from .timer import Timer
from .rprint import rlog as log
from PIL import Image


class XPoseRunner(object):
    def __init__(self, model_config_path, model_checkpoint_path, cpu_only=False, device_id=0):
        self.device = f"cuda:{device_id}" if not cpu_only else "cpu"
        self.model = self.load_animal_model(model_config_path, model_checkpoint_path, self.device)
        self.timer = Timer()
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)

    def load_animal_model(self, model_config_path, model_checkpoint_path, device):
        args = Config.fromfile(model_config_path)
        args.device = device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model.eval()
        return model

    def text_encoding(self, instance_names, keypoints_names):
        ins_text_embeddings = []
        for cat in instance_names:
            instance_description = f"a photo of {cat.lower().replace('_', ' ').replace('-', ' ')}"
            text = clip.tokenize(instance_description).to(self.device)
            text_features = self.clip_model.encode_text(text)
            ins_text_embeddings.append(text_features)
        ins_text_embeddings = torch.cat(ins_text_embeddings, dim=0)

        kpt_text_embeddings = []
        for kpt in keypoints_names:
            kpt_description = f"a photo of {kpt.lower().replace('_', ' ')}"
            text = clip.tokenize(kpt_description).to(self.device)
            with torch.no_grad():
                text_features = self.clip_model.encode_text(text)
            kpt_text_embeddings.append(text_features)
        kpt_text_embeddings = torch.cat(kpt_text_embeddings, dim=0)

        return ins_text_embeddings, kpt_text_embeddings

    def load_image(self, input_image):
        image_pil = input_image.convert("RGB")
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image, _ = transform(image_pil, None)
        return image_pil, image

    def get_unipose_output(self, image, instance_text_prompt, keypoint_text_prompt, box_threshold, IoU_threshold):
        instance_list = instance_text_prompt.split(',')
        ins_text_embeddings, kpt_text_embeddings = self.text_encoding(instance_list, keypoint_text_prompt)
        target = {
            "instance_text_prompt": instance_list,
            "keypoint_text_prompt": keypoint_text_prompt,
            "object_embeddings_text": ins_text_embeddings.float(),
            "kpts_embeddings_text": torch.cat((kpt_text_embeddings.float(), torch.zeros(100 - kpt_text_embeddings.shape[0], 512, device=self.device)), dim=0),
            "kpt_vis_text": torch.cat((torch.ones(kpt_text_embeddings.shape[0], device=self.device), torch.zeros(100 - kpt_text_embeddings.shape[0], device=self.device)), dim=0)
        }

        self.model = self.model.to(self.device)
        image = image.to(self.device)

        with torch.no_grad():
            outputs = self.model(image[None], [target])

        logits = outputs["pred_logits"].sigmoid()[0]
        boxes = outputs["pred_boxes"][0]
        keypoints = outputs["pred_keypoints"][0][:, :2 * len(keypoint_text_prompt)]

        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        keypoints_filt = keypoints.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]
        boxes_filt = boxes_filt[filt_mask]
        keypoints_filt = keypoints_filt[filt_mask]

        keep_indices = nms(box_ops.box_cxcywh_to_xyxy(boxes_filt), logits_filt.max(dim=1)[0], iou_threshold=IoU_threshold)

        filtered_boxes = boxes_filt[keep_indices]
        filtered_keypoints = keypoints_filt[keep_indices]

        return filtered_boxes, filtered_keypoints

    def run_unipose(self, input_image, instance_text_prompt, keypoint_text_example, box_threshold, IoU_threshold):
        if keypoint_text_example in globals():
            keypoint_dict = globals()[keypoint_text_example]
        elif instance_text_prompt in globals():
            keypoint_dict = globals()[instance_text_prompt]
        else:
            keypoint_dict = globals()["animal"]

        keypoint_text_prompt = keypoint_dict.get("keypoints")
        keypoint_skeleton = keypoint_dict.get("skeleton")

        image_pil, image = self.load_image(input_image)
        boxes_filt, keypoints_filt = self.get_unipose_output(image, instance_text_prompt, keypoint_text_prompt, box_threshold, IoU_threshold)

        size = image_pil.size
        H, W = size[1], size[0]
        keypoints_filt = keypoints_filt[0].squeeze(0)
        kp = np.array(keypoints_filt.cpu())
        num_kpts = len(keypoint_text_prompt)
        Z = kp[:num_kpts * 2] * np.array([W, H] * num_kpts)
        Z = Z.reshape(num_kpts * 2)
        x = Z[0::2]
        y = Z[1::2]
        return np.stack((x, y), axis=1)

    def warmup(self):
        self.timer.tic()

        img_rgb = Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
        self.run_unipose(img_rgb,'face', 'animal_face', box_threshold=0.0, IoU_threshold=0.0)

        elapse = self.timer.toc()
        log(f'XPoseRunner warmup time: {elapse:.3f}s')
