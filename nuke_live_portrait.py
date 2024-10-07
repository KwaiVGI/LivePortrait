import logging
import torch
import os
import torch.nn.functional as F
from torch import nn
from src.utils.helper import load_model
from src.utils.camera import get_rotation_matrix
from typing import List, Optional, Tuple, Dict
import numpy as np
import torchvision


try:
    from rich.logging import RichHandler
except ImportError:
    RichHandler = None

# Disable CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR warnings
torch.backends.cudnn.benchmark = True

logging.basicConfig(level=logging.INFO, handlers=[RichHandler()] if RichHandler else None)
LOGGER = logging.getLogger(__name__)

MODEL_CONFIG = {
    "model_params": {
        "appearance_feature_extractor_params": {
            "image_channel": 3,
            "block_expansion": 64,
            "num_down_blocks": 2,
            "max_features": 512,
            "reshape_channel": 32,
            "reshape_depth": 16,
            "num_resblocks": 6,
        },
        "motion_extractor_params": {"num_kp": 21, "backbone": "convnextv2_tiny"},
        "warping_module_params": {
            "num_kp": 21,
            "block_expansion": 64,
            "max_features": 512,
            "num_down_blocks": 2,
            "reshape_channel": 32,
            "estimate_occlusion_map": True,
            "dense_motion_params": {
                "block_expansion": 32,
                "max_features": 1024,
                "num_blocks": 5,
                "reshape_depth": 16,
                "compress": 4,
            },
        },
        "spade_generator_params": {
            "upscale": 2,
            "block_expansion": 64,
            "max_features": 512,
            "num_down_blocks": 2,
        },
        "stitching_retargeting_module_params": {
            "stitching": {
                "input_size": 126,
                "hidden_sizes": [128, 128, 64],
                "output_size": 65,
            },
            "lip": {
                "input_size": 65,
                "hidden_sizes": [128, 128, 64],
                "output_size": 63,
            },
            "eye": {
                "input_size": 66,
                "hidden_sizes": [256, 256, 128, 128, 64],
                "output_size": 63,
            },
        },
    }
}


torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(precision=4, suppress=True)


def file_size(file_path):
    """Get the file size in MB."""
    size_in_bytes = os.path.getsize(file_path)
    return int(size_in_bytes / (1024 * 1024))


#  --- Nuke models ---


class LivePortraitNukeFaceDetection(nn.Module):
    """Live Portrait model for Nuke.

    Detect facial landmarks, then crop, align, and stabilize
    the face for further processing.

    Args:
        face_detection: The face detection model.
        face_alignment: The face alignment model.
        scale: The scale of the face.
    """

    def __init__(self, face_detection, face_alignment, scale=2.3) -> None:
        """Initialize the model."""
        super().__init__()
        self.face_detection = face_detection
        self.face_alignment = face_alignment
        self.fiter_threshold = 0.5
        self.reference_scale = 195
        self.resolution = 256
        self.resize = torchvision.transforms.Resize(
            (256, 256),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            max_size=None,
            antialias=True,
        )

        self.crop_cfg_dsize = 512
        self.crop_cfg_scale = scale
        self.crop_cfg_vx_ratio = 0.0
        self.crop_cfg_vy_ratio = -0.125
        self.crop_cfg_face_index = 0
        self.crop_cfg_face_index_order = "large-small"
        self.crop_cfg_rotate = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        MAIN_FACE_INDEX = 0
        resolution = self.resolution
        b, c, h, w = x.shape
        device = torch.device("cuda") if x.is_cuda else torch.device("cpu")
        x0 = x.clone()

        x = x * 255
        x = x.flip(-3)  # RGB to BGR
        x = x - torch.tensor([104.0, 117.0, 123.0], device=device).view(1, 3, 1, 1)

        olist = self.face_detection(x)  # olist = net(img_batch)  # patched uint8_t overflow error

        for i in range(len(olist) // 2):
            olist[i * 2] = F.softmax(olist[i * 2], dim=1)

        bboxlists = self.get_predictions(olist)
        detected_faces = self.filter_bboxes(bboxlists[0])

        if detected_faces.size(0) == 0:
            print("LivePortrait Face Detection: No faces detected")
            return torch.zeros((b, 4, h, w), device=device)

        d = detected_faces[MAIN_FACE_INDEX]

        d_top_left = float(d[0])
        d_bottom_right = float(d[1])
        d_top_right = float(d[2])
        d_bottom_left = float(d[3])

        center_x = d_top_right - (d_top_right - d_top_left) / 2.0
        center_y = d_bottom_left - (d_bottom_left - d_bottom_right) / 2.0
        center_y = center_y - (d_bottom_left - d_bottom_right) * 0.12

        scale = (d_top_right - d_top_left + d_bottom_left - d_bottom_right) / self.reference_scale

        ul = self.transform(
            torch.tensor([[1, 1]]),
            torch.tensor([[center_x, center_y]], device=device),
            scale,
            resolution,
            True,
        )[0]
        ul_x = int(ul[0])
        ul_y = int(ul[1])

        br = self.transform(
            torch.tensor([[resolution, resolution]]),
            torch.tensor([[center_x, center_y]], device=device),
            scale,
            resolution,
            True,
        )[0]
        br_x = int(br[0])
        br_y = int(br[1])

        crop = torch.zeros([b, c, br_y - ul_y, br_x - ul_x], device=device)
        newX = torch.tensor([max(1, -ul_x + 1), min(br_x, w) - ul_x])
        newY = torch.tensor([max(1, -ul_y + 1), min(br_y, h) - ul_y])

        oldX = torch.tensor([max(1, ul_x + 1), min(br_x, w)])
        oldY = torch.tensor([max(1, ul_y + 1), min(br_y, h)])

        crop[:, :, newY[0] - 1 : newY[1], newX[0] - 1 : newX[1]] = x0[
            :, :, oldY[0] - 1 : oldY[1], oldX[0] - 1 : oldX[1]
        ]

        crop_resized = self.resize(crop)
        fa_out = self.face_alignment(crop_resized)

        pts, pts_img, scores = self.get_preds_fromhm(
            fa_out, torch.tensor([center_x, center_y], device=device), scale
        )

        lmk_tensor = pts_img.squeeze()

        ret_dct = self.crop_image(
            x0.squeeze(),
            lmk_tensor,
            dsize=self.crop_cfg_dsize,
            scale=self.crop_cfg_scale,
            vx_ratio=self.crop_cfg_vx_ratio,
            vy_ratio=self.crop_cfg_vy_ratio,
            flag_do_rot=True,
            use_lip=True,
        )

        ret_dct["cropped_image_256"] = self.resize(ret_dct["img_crop"].permute(2, 0, 1).unsqueeze(0))
        ret_dct["pt_crop_256x256"] = ret_dct["pt_crop"] * 256 / self.crop_cfg_dsize

        # TODO: implement self.landmark_runner.run(img_rgb, pts)

        out = torch.zeros((b, 4, h, w), device=device)  # RGBA
        out[0, :3, 0:256, 0:256] = ret_dct["cropped_image_256"].squeeze()
        out[0, -1, 0, :5] = detected_faces[MAIN_FACE_INDEX].reshape(-1)
        out[0, -1, 1:3, :68] = ret_dct["pt_crop"].permute(1, 0)
        out[0, -1, 3:4, :9] = ret_dct["M_o2c"].reshape(-1)
        out[0, -1, 4:5, :9] = ret_dct["M_c2o"].reshape(-1)

        return out.contiguous()

    # All functions below were adapted from the original code,
    # stripped of any external dependencies.

    def decode(self, loc, priors, variances):
        """Decode locations from predictions using priors to undo
        the encoding we did for offset regression at train time.
        Args:
            loc (tensor): location predictions for loc layers,
                Shape: [num_priors,4]
            priors (tensor): Prior boxes in center-offset form.
                Shape: [num_priors,4].
            variances: (list[float]) Variances of priorboxes
        Return:
            decoded bounding box predictions

        """
        boxes = torch.cat(
            (
                priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1]),
            ),
            1,
        )
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    def get_predictions(self, olist: List[torch.Tensor]):
        variances = torch.tensor([0.1, 0.2], dtype=torch.float32, device=olist[0].device)
        bboxlists = []

        for i in range(len(olist) // 2):
            ocls, oreg = olist[i * 2], olist[i * 2 + 1]
            stride = 2 ** (i + 2)
            mask = ocls[:, 1, :, :] > 0.05
            hindex, windex = torch.where(mask)[1], torch.where(mask)[2]

            for idx in range(hindex.size(0)):
                h = hindex[idx]
                w = windex[idx]
                axc, ayc = stride / 2 + w * stride, stride / 2 + h * stride
                axc_float = float(axc)
                ayx_float = float(ayc)
                priors = torch.tensor(
                    [[axc_float / 1.0, ayx_float / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]],
                    device="cuda",
                )
                score = ocls[:, 1, h, w].unsqueeze(1)
                loc = oreg[:, :, h, w].clone()
                boxes = self.decode(loc, priors, variances)
                bboxlists.append(torch.cat((boxes, score), dim=1))

        if len(bboxlists) == 0:
            output = torch.zeros((1, 0, 5))  # Assuming 5 columns in the final output
        else:
            output = torch.stack(bboxlists, dim=1)

        return output

    def nms(self, dets: torch.Tensor, thresh: float) -> List[int]:
        if dets.size(0) == 0:
            return []

        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = torch.argsort(scores, descending=True)

        keep: List[int] = []

        while order.numel() > 0:
            i = int(order[0])
            keep.append(i)

            if order.size(0) == 1:
                break

            xx1 = torch.max(x1[i], x1[order[1:]])
            yy1 = torch.max(y1[i], y1[order[1:]])
            xx2 = torch.min(x2[i], x2[order[1:]])
            yy2 = torch.min(y2[i], y2[order[1:]])

            w = torch.clamp(xx2 - xx1 + 1, min=0.0)
            h = torch.clamp(yy2 - yy1 + 1, min=0.0)

            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = torch.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

    def filter_bboxes(self, bboxlist):
        nms_thresh = 0.3

        if bboxlist.size(0) > 0:
            keep_indices = self.nms(bboxlist, nms_thresh)
            bboxlist = bboxlist[keep_indices]

            mask = bboxlist[:, -1] > self.fiter_threshold
            bboxlist = bboxlist[mask]

        return bboxlist

    def transform(self, points, center, scale: float, resolution: int, invert: bool = False):
        """Generate and affine transformation matrix.

        Given a set of points, a center, a scale and a targer resolution, the
        function generates and affine transformation matrix. If invert is ``True``
        it will produce the inverse transformation.

        Arguments:
            points -- the input 2D points
            center -- the center around which to perform the transformations
            scale -- the scale of the face/object
            resolution -- the output resolution

        Keyword Arguments:
            invert {bool} -- define wherever the function should produce the direct or the
            inverse transformation matrix (default: {False})
        """
        N = points.shape[0]
        _pt = torch.ones(N, 3, device=points.device)
        _pt[:, 0:2] = points

        h = 200.0 * scale

        t = torch.eye(3, device=points.device).unsqueeze(0).repeat(N, 1, 1)  # [N, 3, 3]
        t[:, 0, 0] = resolution / h
        t[:, 1, 1] = resolution / h
        t[:, 0, 2] = resolution * (-center[:, 0] / h + 0.5)
        t[:, 1, 2] = resolution * (-center[:, 1] / h + 0.5)

        if invert:
            t = torch.inverse(t)

        new_point = torch.bmm(t, _pt.unsqueeze(-1))[:, 0:2, 0]  # [N, 2]

        return new_point.long()

    def _get_preds_fromhm(
        self,
        hm: torch.Tensor,
        idx: torch.Tensor,
        center: Optional[torch.Tensor],
        scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Obtain (x,y) coordinates given a set of N heatmaps and the
        coresponding locations of the maximums. If the center
        and the scale is provided the function will return the points also in
        the original coordinate frame.

        Arguments:
            hm {torch.tensor} -- the predicted heatmaps, of shape [B, N, W, H]

        Keyword Arguments:
            center {torch.tensor} -- the center of the bounding box (default: {None})
            scale {float} -- face scale (default: {None})
        """
        B, C, H, W = hm.shape
        idx = idx + 1

        preds = idx.unsqueeze(-1).repeat(1, 1, 2).float()
        preds[:, :, 0] = (preds[:, :, 0] - 1) % W + 1
        preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / H) + 1

        hm_padded = F.pad(hm, (1, 1, 1, 1), mode="replicate")  # [B, C, H+2, W+2]

        pX = preds[:, :, 0].long() - 1
        pY = preds[:, :, 1].long() - 1

        pX_padded = pX + 1
        pY_padded = pY + 1

        pX_p1 = pX_padded + 1
        pX_m1 = pX_padded - 1
        pY_p1 = pY_padded + 1
        pY_m1 = pY_padded - 1

        B_C = B * C
        hm_padded_flat = hm_padded.contiguous().view(B_C, H + 2, W + 2)
        pX_padded_flat = pX_padded.view(B_C)
        pY_padded_flat = pY_padded.view(B_C)
        pX_p1_flat = pX_p1.view(B_C)
        pX_m1_flat = pX_m1.view(B_C)
        pY_p1_flat = pY_p1.view(B_C)
        pY_m1_flat = pY_m1.view(B_C)

        batch_channel_indices = torch.arange(B_C, device=hm.device)

        hm_padded_flat = hm_padded_flat.float()
        val_pX_p1 = hm_padded_flat[batch_channel_indices, pY_padded_flat, pX_p1_flat]
        val_pX_m1 = hm_padded_flat[batch_channel_indices, pY_padded_flat, pX_m1_flat]
        val_pY_p1 = hm_padded_flat[batch_channel_indices, pY_p1_flat, pX_padded_flat]
        val_pY_m1 = hm_padded_flat[batch_channel_indices, pY_m1_flat, pX_padded_flat]

        diff_x = val_pX_p1 - val_pX_m1
        diff_y = val_pY_p1 - val_pY_m1
        diff = torch.stack([diff_x, diff_y], dim=1)  # Shape [B_C, 2]

        sign_diff = torch.sign(diff)
        preds_flat = preds.view(B_C, 2)
        preds_flat += sign_diff * 0.25

        preds = preds_flat.view(B, C, 2)
        preds -= 0.5

        if center is not None and scale is not None:
            preds_flat = preds.view(B * C, 2)
            if center.dim() == 1:
                center_expanded = center.view(1, 2).expand(B * C, 2)
            else:
                center_expanded = center.view(B, 1, 2).expand(B, C, 2).reshape(B * C, 2)
            preds_orig_flat = self.transform(preds_flat, center_expanded, scale, H, invert=True)
            preds_orig = preds_orig_flat.view(B, C, 2)
        else:
            preds_orig = torch.zeros_like(preds)

        return preds, preds_orig

    def get_preds_fromhm(
        self, hm: torch.Tensor, center: torch.Tensor, scale: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Obtain (x,y) coordinates given a set of N heatmaps. If the center
        and the scale is provided the function will return the points also in
        the original coordinate frame.

        Arguments:
            hm {torch.tensor} -- the predicted heatmaps, of shape [B, N, W, H]

        Keyword Arguments:
            center {torch.tensor} -- the center of the bounding box (default: {None})
            scale {float} -- face scale (default: {None})
        """
        B, C, H, W = hm.shape
        hm_reshape = hm.view(B, C, H * W)
        idx = torch.argmax(hm_reshape, dim=-1)
        scores = torch.gather(hm_reshape, dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)
        preds, preds_orig = self._get_preds_fromhm(hm, idx, center, scale)
        return preds, preds_orig, scores

    def crop_image(
        self,
        img: torch.Tensor,
        pts: torch.Tensor,
        dsize: int = 224,
        scale: float = 1.5,
        vx_ratio: float = 0.0,
        vy_ratio: float = -0.1,
        flag_do_rot: bool = True,
        use_lip: bool = True,
    ) -> Dict[str, torch.Tensor]:
        pts = pts.float()
        M_INV, _ = self._estimate_similar_transform_from_pts(
            pts,
            dsize=dsize,
            scale=scale,
            vx_ratio=vx_ratio,
            vy_ratio=vy_ratio,
            flag_do_rot=flag_do_rot,
            use_lip=use_lip,
        )

        img_crop = self._transform_img(img, M_INV, dsize)
        pt_crop = self._transform_pts(pts, M_INV)

        M_o2c = torch.vstack(
            [M_INV, torch.tensor([0, 0, 1], dtype=M_INV.dtype, device=M_INV.device)]
        )
        M_c2o = torch.inverse(M_o2c)

        ret_dct = {
            "M_o2c": M_o2c,
            "M_c2o": M_c2o,
            "img_crop": img_crop,
            "pt_crop": pt_crop,
        }

        return ret_dct

    def _estimate_similar_transform_from_pts(
        self,
        pts: torch.Tensor,
        dsize: int,
        scale: float = 1.5,
        vx_ratio: float = 0.0,
        vy_ratio: float = -0.1,
        flag_do_rot: bool = True,
        use_lip: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate Similar Transform from Points

        Calculate the affine matrix of the cropped image from sparse points,
        the original image to the cropped image, the inverse is the cropped image to the original image

        pts: landmark, 101 or 68 points or other points, Nx2
        scale: the larger scale factor, the smaller face ratio
        vx_ratio: x shift
        vy_ratio: y shift, the smaller the y shift, the lower the face region
        rot_flag: if it is true, conduct correction
        """
        center, size, angle = self.parse_rect_from_landmark(
            pts, scale=scale, vx_ratio=vx_ratio, vy_ratio=vy_ratio, use_lip=use_lip
        )

        s = dsize / size[0]
        tgt_center = torch.tensor([dsize / 2.0, dsize / 2.0], dtype=pts.dtype)

        if flag_do_rot:
            costheta = torch.cos(angle)
            sintheta = torch.sin(angle)
            cx, cy = center[0], center[1]
            tcx, tcy = tgt_center[0], tgt_center[1]

            M_INV = torch.zeros((2, 3), dtype=pts.dtype, device=pts.device)
            M_INV[0, 0] = s * costheta
            M_INV[0, 1] = s * sintheta
            M_INV[0, 2] = tcx - s * (costheta * cx + sintheta * cy)
            M_INV[1, 0] = -s * sintheta
            M_INV[1, 1] = s * costheta
            M_INV[1, 2] = tcy - s * (-sintheta * cx + costheta * cy)
        else:
            M_INV = torch.zeros((2, 3), dtype=pts.dtype, device=pts.device)
            M_INV[0, 0] = s
            M_INV[0, 1] = 0.0
            M_INV[0, 2] = tgt_center[0] - s * center[0]
            M_INV[1, 0] = 0.0
            M_INV[1, 1] = s
            M_INV[1, 2] = tgt_center[1] - s * center[1]

        M_INV_H = torch.cat(
            [M_INV, torch.tensor([[0, 0, 1]], dtype=M_INV.dtype, device=pts.device)], dim=0
        )
        M = torch.inverse(M_INV_H)

        return M_INV, M[:2, :]

    def parse_rect_from_landmark(
        self,
        pts: torch.Tensor,
        scale: float = 1.5,
        need_square: bool = True,
        vx_ratio: float = 0.0,
        vy_ratio: float = 0.0,
        use_deg_flag: bool = False,
        use_lip: bool = True,
    ):
        """Parse center, size, angle from 101/68/5/x landmarks

        vx_ratio: the offset ratio along the pupil axis x-axis, multiplied by size
        vy_ratio: the offset ratio along the pupil axis y-axis, multiplied by size, which is used to contain more forehead area
        """
        pt2 = self.parse_pt2_from_pt68(pts, use_lip=use_lip)

        if not use_lip:
            v = pt2[1] - pt2[0]
            new_pt1 = torch.stack([pt2[0, 0] - v[1], pt2[0, 1] + v[0]], dim=0)
            pt2 = torch.stack([pt2[0], new_pt1], dim=0)

        uy = pt2[1] - pt2[0]
        l = torch.norm(uy)
        if l.item() <= 1e-3:
            uy = torch.tensor([0.0, 1.0], dtype=pts.dtype)
        else:
            uy = uy / l
        ux = torch.stack((uy[1], -uy[0]))

        angle = torch.acos(ux[0])
        if ux[1].item() < 0:
            angle = -angle

        M = torch.stack([ux, uy], dim=0)

        center0 = torch.mean(pts, dim=0)
        rpts = torch.matmul(pts - center0, M.T)
        lt_pt = torch.min(rpts, dim=0)[0]
        rb_pt = torch.max(rpts, dim=0)[0]
        center1 = (lt_pt + rb_pt) / 2

        size = rb_pt - lt_pt
        if need_square:
            m = torch.max(size[0], size[1])
            size = torch.stack([m, m])

        size = size * scale
        center = center0 + ux * center1[0] + uy * center1[1]
        center = center + ux * (vx_ratio * size) + uy * (vy_ratio * size)

        if use_deg_flag:
            angle = torch.rad2deg(angle)

        return center, size, angle

    def parse_pt2_from_pt68(self, pt68: torch.Tensor, use_lip: bool = True) -> torch.Tensor:
        if use_lip:
            left_eye = pt68[42:48].mean(dim=0)
            right_eye = pt68[36:42].mean(dim=0)
            mouth_center = (pt68[48] + pt68[54]) / 2.0

            pt68_new = torch.stack([left_eye, right_eye, mouth_center], dim=0)
            pt2 = torch.stack([(pt68_new[0] + pt68_new[1]) / 2.0, pt68_new[2]], dim=0)
        else:
            left_eye = pt68[42:48].mean(dim=0)
            right_eye = pt68[36:42].mean(dim=0)

            pt2 = torch.stack([left_eye, right_eye], dim=0)

            v = pt2[1] - pt2[0]
            pt2[1, 0] = pt2[0, 0] - v[1]
            pt2[1, 1] = pt2[0, 1] + v[0]

        return pt2

    def _transform_img(self, img: torch.Tensor, M: torch.Tensor, dsize: int):
        """conduct similarity or affine transformation to the image, do not do border operation!
        img:
        M: 2x3 matrix or 3x3 matrix
        dsize: target shape (width, height)
        """
        if isinstance(dsize, (tuple, list)):
            out_h, out_w = dsize
        else:
            out_h = out_w = dsize

        C, H_in, W_in = img.shape

        M_norm = self._normalize_affine(M, W_in, H_in, out_w, out_h)
        grid = F.affine_grid(M_norm.unsqueeze(0), [1, C, out_h, out_w], align_corners=False)
        img = img.unsqueeze(0)

        img_warped = F.grid_sample(
            img, grid, align_corners=False, mode="bilinear", padding_mode="zeros"
        )
        img_warped = img_warped.squeeze(0)
        img_warped = img_warped.permute(1, 2, 0)  # [H, W, C]

        return img_warped

    def _normalize_affine(self, M: torch.Tensor, W_in: int, H_in: int, W_out: int, H_out: int):
        device = M.device
        dtype = M.dtype

        M_h = torch.cat([M, torch.tensor([[0.0, 0.0, 1.0]], dtype=dtype, device=device)], dim=0)

        M_h_inv = torch.inverse(M_h)

        W_in_f = float(W_in)
        H_in_f = float(H_in)
        W_out_f = float(W_out)
        H_out_f = float(H_out)

        S_in = torch.zeros(3, 3, dtype=dtype, device=device)
        S_in[0, 0] = 2.0 / W_in_f
        S_in[0, 1] = 0.0
        S_in[0, 2] = -1.0
        S_in[1, 0] = 0.0
        S_in[1, 1] = 2.0 / H_in_f
        S_in[1, 2] = -1.0
        S_in[2, 0] = 0.0
        S_in[2, 1] = 0.0
        S_in[2, 2] = 1.0

        S_out = torch.zeros(3, 3, dtype=dtype, device=device)
        S_out[0, 0] = W_out_f / 2.0
        S_out[0, 1] = 0.0
        S_out[0, 2] = W_out_f / 2.0
        S_out[1, 0] = 0.0
        S_out[1, 1] = H_out_f / 2.0
        S_out[1, 2] = H_out_f / 2.0
        S_out[2, 0] = 0.0
        S_out[2, 1] = 0.0
        S_out[2, 2] = 1.0

        M_combined = torch.matmul(torch.matmul(S_in, M_h_inv), S_out)

        return M_combined[:2, :]

    def _transform_pts(self, pts: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        return torch.matmul(pts, M[:, :2].T) + M[:, 2]


class LivePortraitNukeAppearanceFeatureExtractor(nn.Module):
    """Live Portrait model for Nuke.

    Args:
        encoder: The encoder model.
        decoder: The decoder model.
        n: Depth Anything window list parameter.
    """

    def __init__(self, model) -> None:
        """Initialize the model."""
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        out = self.model(x)  # Tensor[1, 32, 16, 64, 64]

        # Split batch by 2 as 16 rows in the red and green channels
        out_block = out.view([1, 2, 16, 16, 64, 64])
        out_block = out_block.permute(0, 1, 2, 4, 3, 5).reshape(1, 2, 16 * 64, 64 * 16)
        return out_block


class LivePortraitNukeMotionExtractor(nn.Module):
    """LivePortraitNukeMotionExtractor model for Nuke.

    Args:
        model: The encoder model.
        n: Depth Anything window list parameter.
    """

    def __init__(self, model) -> None:
        """Initialize the model."""
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        kp_info = self.model(x)

        for k, v in kp_info.items():
            if isinstance(v, torch.Tensor):
                kp_info[k] = v.float()

        bs = kp_info["kp"].shape[0]

        pitch = self.headpose_pred_to_degree(kp_info["pitch"])  # Bx1
        yaw = self.headpose_pred_to_degree(kp_info["yaw"])  # Bx1
        roll = self.headpose_pred_to_degree(kp_info["roll"])  # Bx1
        rot_mat = get_rotation_matrix(pitch, yaw, roll)  # Bx3x3
        exp = kp_info["exp"]  # .reshape(bs, -1, 3)  # BxNx3
        kp = kp_info["kp"]  # .reshape(bs, -1, 3)  # BxNx3
        scale = kp_info["scale"]
        t = kp_info["t"]

        if kp.ndim == 2:
            num_kp = kp.shape[1] // 3  # Bx(num_kpx3)
        else:
            num_kp = kp.shape[1]  # Bxnum_kpx3

        kp_transformed = kp.view(bs, num_kp, 3) @ rot_mat + exp.view(bs, num_kp, 3)
        kp_transformed *= scale[..., None]  # (bs, k, 3) * (bs, 1, 1) = (bs, k, 3)
        kp_transformed[:, :, 0:2] += t[:, None, 0:2]  # remove z, only apply tx ty

        out = torch.zeros([b, 1, h, w], device=x.device)
        out[0, 0, 0, :3] = torch.cat([pitch, yaw, roll], dim=0)
        out[0, 0, 1, :9] = rot_mat.reshape(-1)
        out[0, 0, 2, :63] = kp.reshape(-1)
        out[0, 0, 3, :63] = kp_transformed.reshape(-1)
        out[0, 0, 4, :63] = exp.reshape(-1)
        out[0, 0, 5, :1] = scale.reshape(-1)
        out[0, 0, 6, :3] = t.reshape(-1)

        return out.contiguous()

    def headpose_pred_to_degree(self, x: torch.Tensor) -> torch.Tensor:
        idx_tensor = torch.arange(0, 66, device=x.device, dtype=torch.float32)
        pred = F.softmax(x, dim=1)
        degree = torch.sum(pred * idx_tensor, dim=1) * 3 - 97.5
        return degree


class LivePortraitNukeWarpingModule(nn.Module):
    """LivePortraitNukeMotionExtractor model for Nuke.

    Args:
        model: The encoder model.
        n: Depth Anything window list parameter.
    """

    def __init__(self, model) -> None:
        """Initialize the model."""
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input tensor [1,2,1040,1024] - 2 channels image, 1040x1024
        # Split the tensor x into feature_3d, kp_source, and kp_driving
        kp_source = x[:, 2, 0, :63].reshape(1, 21, 3).contiguous()
        kp_driving = x[:, 2, 1, :63].reshape(1, 21, 3).contiguous()
        feature_3d = x[:, :2, :, :]  # .reshape(1, 32, 16, 64, 64)
        feature_3d = (
            feature_3d.view(1, 2, 16, 64, 16, 64)
            .permute(0, 1, 2, 4, 3, 5)
            .contiguous()
            .view(1, 32, 16, 64, 64)
            .contiguous()
        )

        out_dct = self.model(feature_3d, kp_driving, kp_source)
        out = out_dct["out"]

        assert out is not None

        out = out.view(1, 16, 16, 64, 64)
        out = out.permute(0, 1, 3, 2, 4)
        out = out.reshape(1, 1, 1024, 1024)
        return out.contiguous()


class LivePortraitNukeSpadeGenerator(nn.Module):
    """LivePortraitNukeMotionExtractor model for Nuke.

    Args:
        model: The encoder model.
        n: Depth Anything window list parameter.
    """

    def __init__(self, model) -> None:
        """Initialize the model."""
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        # Input tensor [1, 1, 1040,1024] to [1, 256, 64, 64]
        x = x.view(1, 16, 64, 16, 64)
        x = x.permute(0, 1, 3, 2, 4).reshape(1, 256, 64, 64)
        out = self.model(feature=x)
        out = out.contiguous()
        return out


class LivePortraitNukeStitchingModule(nn.Module):
    """LivePortraitNukeMotionExtractor model for Nuke.

    Args:
        model: The encoder model.
        n: Depth Anything window list parameter.
    """

    def __init__(self, model) -> None:
        """Initialize the model."""
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        kp_source = x[:, 0, 0, :63].reshape(1, 21, 3).contiguous()  # 1x21x3
        kp_driving = x[:, 0, 1, :63].reshape(1, 21, 3).contiguous()  # 1x21x3
        kp_driving_new = kp_driving.clone()

        bs, num_kp = kp_source.shape[:2]

        feat_stiching = self.concat_feat(kp_source, kp_driving)
        delta = self.model(feat_stiching)  # 1x65
        delta_exp = delta[..., : 3 * num_kp].reshape(bs, num_kp, 3)  # 1x21x3
        delta_tx_ty = delta[..., 3 * num_kp : 3 * num_kp + 2].reshape(bs, 1, 2)  # 1x1x2

        kp_driving_new += delta_exp
        kp_driving_new[..., :2] += delta_tx_ty

        out = torch.zeros([b, 1, h, w], device=x.device)
        out[0, 0, 0, :63] = kp_driving_new.reshape(-1)

        return out.contiguous()

    def concat_feat(self, kp_source: torch.Tensor, kp_driving: torch.Tensor) -> torch.Tensor:
        """
        kp_source: (bs, k, 3)
        kp_driving: (bs, k, 3)
        Return: (bs, 2k*3)
        """
        bs_src = kp_source.shape[0]
        bs_dri = kp_driving.shape[0]
        assert bs_src == bs_dri, "batch size must be equal"

        feat = torch.cat([kp_source.view(bs_src, -1), kp_driving.view(bs_dri, -1)], dim=1)
        return feat


#  --- Tracing original models ---


def face_detection():
    """
    SFDDetector = getattr(__import__('custom_nodes.ComfyUI-LivePortraitKJ.face_alignment.detection.sfd.sfd_detector', fromlist=['']), 'SFDDetector')
    sfd_detector = SFDDetector("cuda")
    sf3d = sfd_detector.face_detector
    sfd_detector_traced = torch.jit.script(sfd_detector.face_detector)
    sfd_detector_traced.save("sfd_detector_traced.pt")
    """
    sfd_detector_traced = torch.load("./pretrained_weights/sfd_detector_traced.pt")
    return sfd_detector_traced


def trace_appearance_feature_extractor():
    LOGGER.info("--- Tracing appearance_feature_extractor ---")
    appearance_feature_extractor = load_model(
        ckpt_path="./pretrained_weights/liveportrait/base_models/appearance_feature_extractor.pth",
        model_config=MODEL_CONFIG,
        device=0,
        model_type="appearance_feature_extractor",
    )

    with torch.no_grad():
        appearance_feature_extractor.eval()
        appearance_feature_extractor = torch.jit.script(appearance_feature_extractor)

    LOGGER.info("Traced appearance_feature_extractor")

    destination = "./build/appearance_feature_extractor.pt"
    torch.jit.save(appearance_feature_extractor, destination)
    LOGGER.info("Model saved to: %s", destination)

    return appearance_feature_extractor


def trace_motion_extractor():
    LOGGER.info("--- Tracing motion_extractor ---")

    motion_extractor = load_model(
        ckpt_path="./pretrained_weights/liveportrait/base_models/motion_extractor.pth",
        model_config=MODEL_CONFIG,
        device=0,
        model_type="motion_extractor",
    )

    with torch.no_grad():
        motion_extractor.eval()
        motion_extractor = torch.jit.script(motion_extractor)

    LOGGER.info("Traced motion_extractor")

    destination = "./build/motion_extractor.pt"
    torch.jit.save(motion_extractor, destination)
    LOGGER.info("Model saved to: %s", destination)

    return motion_extractor


def trace_warping_module():
    LOGGER.info("--- Tracing warping_module ---")

    warping_module = load_model(
        ckpt_path="./pretrained_weights/liveportrait/base_models/warping_module.pth",
        model_config=MODEL_CONFIG,
        device=0,
        model_type="warping_module",
    )

    with torch.no_grad():
        warping_module.eval()
        warping_module = torch.jit.script(warping_module)

    LOGGER.info("Traced warping_module")

    destination = "./build/warping_module.pt"
    torch.jit.save(warping_module, destination)
    LOGGER.info("Model saved to: %s", destination)

    return warping_module


def trace_spade_generator():
    LOGGER.info("--- Tracing spade_generator ---")

    spade_generator = load_model(
        ckpt_path="./pretrained_weights/liveportrait/base_models/spade_generator.pth",
        model_config=MODEL_CONFIG,
        device=0,
        model_type="spade_generator",
    )

    with torch.no_grad():
        spade_generator.eval()
        spade_generator = torch.jit.script(spade_generator)

    LOGGER.info("Traced spade_generator")

    destination = "./build/spade_generator.pt"
    torch.jit.save(spade_generator, destination)
    LOGGER.info("Model saved to: %s", destination)

    return spade_generator


def trace_stitching_retargeting_module():
    LOGGER.info("--- Tracing stitching_retargeting_module ---")

    stitching_retargeting_module = load_model(
        ckpt_path="./pretrained_weights/liveportrait/retargeting_models/stitching_retargeting_module.pth",
        model_config=MODEL_CONFIG,
        device=0,
        model_type="stitching_retargeting_module",
    )

    with torch.no_grad():
        stitching = stitching_retargeting_module["stitching"].eval()
        lip = stitching_retargeting_module["lip"].eval()
        eye = stitching_retargeting_module["eye"].eval()

        stitching_trace = torch.jit.script(stitching)
        lip_trace = torch.jit.script(lip)
        eye_trace = torch.jit.script(eye)

    LOGGER.info("Traced stitching_retargeting_module")

    destination = "./build/stitching_retargeting_module_stitching.pt"
    torch.jit.save(stitching_trace, destination)

    destination = "./build/stitching_retargeting_module_eye.pt"
    torch.jit.save(eye_trace, destination)

    destination = "./build/stitching_retargeting_module_lip.pt"
    torch.jit.save(lip_trace, destination)

    LOGGER.info("Model saved to: %s", destination)

    return stitching_trace, lip_trace, eye_trace


# --- Tracing Nuke models ---


def trace_face_detection_nuke(run_test=False):
    sf3d_face_detection = torch.jit.load("./pretrained_weights/sfd_detector_traced.pt").cuda()
    face_alignment = torch.jit.load(
        "./pretrained_weights/from_kj/2DFAN4-cd938726ad.zip"
    ).cuda()

    model = LivePortraitNukeFaceDetection(
        face_detection=sf3d_face_detection, face_alignment=face_alignment
    )

    def test_forward():
        with torch.no_grad():
            m = torch.randn(1, 3, 720, 1280).cuda()
            model.eval()
            out = model(m)
            LOGGER.info(out.shape)

    if run_test:
        test_forward()

    model_traced = torch.jit.script(model)
    destination = "./build/face_detection_nuke.pt"
    torch.jit.save(model_traced, destination)

    LOGGER.info("Model saved to: %s", destination)


def trace_appearance_feature_extractor_nuke(run_test=False):
    appearance_feature_extractor = trace_appearance_feature_extractor()
    model = LivePortraitNukeAppearanceFeatureExtractor(model=appearance_feature_extractor)

    def test_forward():
        with torch.no_grad():
            m = torch.randn(1, 3, 256, 256).cuda()
            model.eval()
            out = model(m)
            LOGGER.info(out.shape)

    if run_test:
        test_forward()

    model_traced = torch.jit.script(model)

    destination = "./build/appearance_feature_extractor_nuke.pt"
    torch.jit.save(model_traced, destination)
    LOGGER.info("Model saved to: %s", destination)


def trace_motion_extractor_nuke(run_test=False):
    motion_extractor = trace_motion_extractor()
    model = LivePortraitNukeMotionExtractor(model=motion_extractor)
    model_traced = torch.jit.script(model)

    def test_forward():
        with torch.no_grad():
            m = torch.randn(1, 3, 256, 256).cuda()
            model.eval()
            out = model(m)
            LOGGER.info(out.shape)

    if run_test:
        test_forward()

    destination = "./build/motion_extractor_nuke.pt"
    torch.jit.save(model_traced, destination)
    LOGGER.info("Model saved to: %s", destination)
    return model_traced


def trace_warping_module_nuke(run_test=False):
    warping_module = trace_warping_module()
    model = LivePortraitNukeWarpingModule(model=warping_module)

    def test_forward():
        with torch.no_grad():
            m = torch.randn([1, 3, 1024, 1024], dtype=torch.float32, device="cuda")
            model.eval()
            out = model(m)
            LOGGER.info(out.shape)

    if run_test:
        test_forward()

    model_traced = torch.jit.script(model)
    destination = "./build/warping_module_nuke.pt"
    torch.jit.save(model_traced, destination)
    LOGGER.info("Model saved to: %s", destination)


def trace_spade_generator_nuke(run_test=False):
    warping_module = trace_spade_generator()
    model = LivePortraitNukeSpadeGenerator(model=warping_module)

    def test_forward():
        with torch.no_grad():
            m = torch.randn([1, 1, 1024, 1024], dtype=torch.float32, device="cuda")
            model.eval()
            out = model(m)
            LOGGER.info(out.shape)

    if run_test:
        test_forward()

    model_traced = torch.jit.script(model)
    destination = "./build/spade_generator_nuke.pt"
    torch.jit.save(model_traced, destination)
    LOGGER.info("Model saved to: %s", destination)


def trace_stitching_retargeting_module_nuke(run_test=False):
    stitching_model, lip_model, eye_model = trace_stitching_retargeting_module()

    model = LivePortraitNukeStitchingModule(model=stitching_model)

    def test_forward():
        with torch.no_grad():
            m = torch.randn([1, 1, 64, 64], dtype=torch.float32, device="cuda")
            model.eval()
            out = model(m)
            LOGGER.info(out.shape)

    if run_test:
        test_forward()

    model_traced = torch.jit.script(model)
    destination = "./build/stitching_retargeting_module_stitching_nuke.pt"
    torch.jit.save(model_traced, destination)
    LOGGER.info("Model saved to: %s", destination)


if __name__ == "__main__":

    run_test = True
    trace_face_detection_nuke(run_test)
    trace_appearance_feature_extractor_nuke(run_test)
    trace_motion_extractor_nuke(run_test)
    trace_warping_module_nuke(run_test)
    trace_spade_generator_nuke(run_test)
    trace_stitching_retargeting_module_nuke(run_test)
