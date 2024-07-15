# coding: utf-8

"""
The module that predicting a dense motion from sparse motion representation given by kp_source and kp_driving
"""

from torch import nn
import torch.nn.functional as F
import torch
from .util import Hourglass, make_coordinate_grid, kp2gaussian


class DenseMotionNetwork(nn.Module):
    def __init__(self, block_expansion, num_blocks, max_features, num_kp, feature_channel, reshape_depth, compress, estimate_occlusion_map=True):
        super(DenseMotionNetwork, self).__init__()
        self.hourglass = Hourglass(block_expansion=block_expansion, in_features=(num_kp+1)*(compress+1), max_features=max_features, num_blocks=num_blocks)  # ~60+G

        self.mask = nn.Conv3d(self.hourglass.out_filters, num_kp + 1, kernel_size=7, padding=3)  # 65G! NOTE: computation cost is large
        self.compress = nn.Conv3d(feature_channel, compress, kernel_size=1)  # 0.8G
        self.norm = nn.BatchNorm3d(compress, affine=True)
        self.num_kp = num_kp
        self.flag_estimate_occlusion_map = estimate_occlusion_map

        if self.flag_estimate_occlusion_map:
            self.occlusion = nn.Conv2d(self.hourglass.out_filters*reshape_depth, 1, kernel_size=7, padding=3)
        else:
            self.occlusion = None

    def create_sparse_motions(self, feature, kp_driving, kp_source):
        bs, _, d, h, w = feature.shape  # (bs, 4, 16, 64, 64)
        identity_grid = make_coordinate_grid((d, h, w), ref=kp_source)  # (16, 64, 64, 3)
        identity_grid = identity_grid.view(1, 1, d, h, w, 3)  # (1, 1, d=16, h=64, w=64, 3)
        coordinate_grid = identity_grid - kp_driving.view(bs, self.num_kp, 1, 1, 1, 3)

        k = coordinate_grid.shape[1]

        # NOTE: there lacks an one-order flow
        driving_to_source = coordinate_grid + kp_source.view(bs, self.num_kp, 1, 1, 1, 3)    # (bs, num_kp, d, h, w, 3)

        # adding background feature
        identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1, 1)
        sparse_motions = torch.cat([identity_grid, driving_to_source], dim=1)  # (bs, 1+num_kp, d, h, w, 3)
        return sparse_motions

    def create_deformed_feature(self, feature, sparse_motions):
        bs, _, d, h, w = feature.shape
        feature_repeat = feature.unsqueeze(1).unsqueeze(1).repeat(1, self.num_kp+1, 1, 1, 1, 1, 1)      # (bs, num_kp+1, 1, c, d, h, w)
        feature_repeat = feature_repeat.view(bs * (self.num_kp+1), -1, d, h, w)                         # (bs*(num_kp+1), c, d, h, w)
        sparse_motions = sparse_motions.view((bs * (self.num_kp+1), d, h, w, -1))                       # (bs*(num_kp+1), d, h, w, 3)
        sparse_deformed = F.grid_sample(feature_repeat, sparse_motions, align_corners=False)
        sparse_deformed = sparse_deformed.view((bs, self.num_kp+1, -1, d, h, w))                        # (bs, num_kp+1, c, d, h, w)

        return sparse_deformed

    def create_heatmap_representations(self, feature, kp_driving, kp_source):
        spatial_size = feature.shape[3:]  # (d=16, h=64, w=64)
        gaussian_driving = kp2gaussian(kp_driving, spatial_size=spatial_size, kp_variance=0.01)  # (bs, num_kp, d, h, w)
        gaussian_source = kp2gaussian(kp_source, spatial_size=spatial_size, kp_variance=0.01)  # (bs, num_kp, d, h, w)
        heatmap = gaussian_driving - gaussian_source  # (bs, num_kp, d, h, w)

        # adding background feature
        zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1], spatial_size[2]).type(heatmap.dtype).to(heatmap.device)
        heatmap = torch.cat([zeros, heatmap], dim=1)
        heatmap = heatmap.unsqueeze(2)         # (bs, 1+num_kp, 1, d, h, w)
        return heatmap

    def forward(self, feature, kp_driving, kp_source):
        bs, _, d, h, w = feature.shape  # (bs, 32, 16, 64, 64)

        feature = self.compress(feature)  # (bs, 4, 16, 64, 64)
        feature = self.norm(feature)  # (bs, 4, 16, 64, 64)
        feature = F.relu(feature)  # (bs, 4, 16, 64, 64)

        out_dict = dict()

        # 1. deform 3d feature
        sparse_motion = self.create_sparse_motions(feature, kp_driving, kp_source)  # (bs, 1+num_kp, d, h, w, 3)
        deformed_feature = self.create_deformed_feature(feature, sparse_motion)  # (bs, 1+num_kp, c=4, d=16, h=64, w=64)

        # 2. (bs, 1+num_kp, d, h, w)
        heatmap = self.create_heatmap_representations(deformed_feature, kp_driving, kp_source)  # (bs, 1+num_kp, 1, d, h, w)

        input = torch.cat([heatmap, deformed_feature], dim=2)  # (bs, 1+num_kp, c=5, d=16, h=64, w=64)
        input = input.view(bs, -1, d, h, w)  # (bs, (1+num_kp)*c=105, d=16, h=64, w=64)

        prediction = self.hourglass(input)

        mask = self.mask(prediction)
        mask = F.softmax(mask, dim=1)  # (bs, 1+num_kp, d=16, h=64, w=64)
        out_dict['mask'] = mask
        mask = mask.unsqueeze(2)                                   # (bs, num_kp+1, 1, d, h, w)
        sparse_motion = sparse_motion.permute(0, 1, 5, 2, 3, 4)    # (bs, num_kp+1, 3, d, h, w)
        deformation = (sparse_motion * mask).sum(dim=1)            # (bs, 3, d, h, w)  mask take effect in this place
        deformation = deformation.permute(0, 2, 3, 4, 1)           # (bs, d, h, w, 3)

        out_dict['deformation'] = deformation

        if self.flag_estimate_occlusion_map:
            bs, _, d, h, w = prediction.shape
            prediction_reshape = prediction.view(bs, -1, h, w)
            occlusion_map = torch.sigmoid(self.occlusion(prediction_reshape))  # Bx1x64x64
            out_dict['occlusion_map'] = occlusion_map

        return out_dict
