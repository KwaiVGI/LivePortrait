# ------------------------------------------------------------------------
# ED-Pose
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
import os
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from util.misc import inverse_sigmoid
from .ops.modules import MSDeformAttn

from .utils import sigmoid_focal_loss, MLP, _get_activation_fn, gen_sineembed_for_position

class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300,
                 use_dab=False, high_dim_query_update=False, no_sine_embed=False):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.use_dab = use_dab

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec,
                                                            use_dab=use_dab, d_model=d_model, high_dim_query_update=high_dim_query_update, no_sine_embed=no_sine_embed)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            if not self.use_dab:
                self.reference_points = nn.Linear(d_model, 2)

        self.high_dim_query_update = high_dim_query_update
        if high_dim_query_update:
            assert not self.use_dab, "use_dab must be True"

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage and not self.use_dab:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None):
        """
        Input:
            - srcs: List([bs, c, h, w])
            - masks: List([bs, h, w])
        """
        assert self.two_stage or query_embed is not None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2)                # bs, hw, c
            mask = mask.flatten(1)                              # bs, hw
            pos_embed = pos_embed.flatten(2).transpose(1, 2)    # bs, hw, c
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)     # bs, \sum{hxw}, c
        mask_flatten = torch.cat(mask_flatten, 1)   # bs, \sum{hxw}
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)
        # import ipdb; ipdb.set_trace()

        # prepare input for decoder
        bs, _, c = memory.shape
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        elif self.use_dab:
            reference_points = query_embed[..., self.d_model:].sigmoid()
            tgt = query_embed[..., :self.d_model]
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            init_reference_out = reference_points
        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_embed).sigmoid()
                # bs, num_quires, 2
            init_reference_out = reference_points

        # decoder
        # import ipdb; ipdb.set_trace()
        hs, inter_references = self.decoder(tgt, reference_points, memory,
                                            spatial_shapes, level_start_index, valid_ratios,
                                            query_pos=query_embed if not self.use_dab else None,
                                            src_padding_mask=mask_flatten)

        inter_references_out = inter_references
        if self.two_stage:
            return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact
        return hs, init_reference_out, inter_references_out, None, None


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 add_channel_attention=False,
                 use_deformable_box_attn=False,
                 box_attn_type='roi_align',
                 ):
        super().__init__()

        # self attention
        if use_deformable_box_attn:
            self.self_attn = MSDeformableBoxAttention(d_model, n_levels, n_heads, n_boxes=n_points, used_func=box_attn_type)
        else:
            self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # channel attention
        self.add_channel_attention = add_channel_attention
        if add_channel_attention:
            self.activ_channel = _get_activation_fn('dyrelu', d_model=d_model)
            self.norm_channel = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, key_padding_mask=None):
        # self attention
        # import ipdb; ipdb.set_trace()
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        # channel attn
        if self.add_channel_attention:
            src = self.norm_channel(src + self.activ_channel(src))

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        if num_layers > 0:
            self.layers = _get_clones(encoder_layer, num_layers)
        else:
            self.layers = []
            del encoder_layer
        self.num_layers = num_layers
        self.norm = norm

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        """
        Input:
            - src: [bs, sum(hi*wi), 256]
            - spatial_shapes: h,w of each level [num_level, 2]
            - level_start_index: [num_level] start point of level in sum(hi*wi).
            - valid_ratios: [bs, num_level, 2]
            - pos: pos embed for src. [bs, sum(hi*wi), 256]
            - padding_mask: [bs, sum(hi*wi)]
        Intermedia:
            - reference_points: [bs, sum(hi*wi), num_lebel, 2]
        """
        output = src
        # bs, sum(hi*wi), 256
        # import ipdb; ipdb.set_trace()
        if self.num_layers > 0:
            reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 use_deformable_box_attn=False,
                 box_attn_type='roi_align',
                 key_aware_type=None,
                 decoder_sa_type='ca',
                 module_seq=['sa', 'ca', 'ffn'],
                 ):
        super().__init__()
        self.module_seq = module_seq
        assert sorted(module_seq) == ['ca', 'ffn', 'sa']

        # cross attention
        # self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        if use_deformable_box_attn:
            self.cross_attn = MSDeformableBoxAttention(d_model, n_levels, n_heads, n_boxes=n_points, used_func=box_attn_type)
        else:
            self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn, batch_dim=1)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.key_aware_type = key_aware_type
        self.key_aware_proj = None
        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ['sa', 'ca_label', 'ca_content']

        if decoder_sa_type == 'ca_content':
            self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)




    def rm_self_attn_modules(self):
        self.self_attn = None
        self.dropout2 = None
        self.norm2 = None


    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_sa(self,
                # for tgt
                tgt: Optional[Tensor],  # nq, bs, d_model
                tgt_query_pos: Optional[Tensor] = None, # pos for query. MLP(Sine(pos))
                tgt_query_sine_embed: Optional[Tensor] = None, # pos for query. Sine(pos)
                tgt_key_padding_mask: Optional[Tensor] = None,
                tgt_reference_points: Optional[Tensor] = None, # nq, bs, 4

                # for memory
                memory: Optional[Tensor] = None, # hw, bs, d_model
                memory_key_padding_mask: Optional[Tensor] = None,
                memory_level_start_index: Optional[Tensor] = None, # num_levels
                memory_spatial_shapes: Optional[Tensor] = None, # bs, num_levels, 2
                memory_pos: Optional[Tensor] = None, # pos for memory

                # sa
                self_attn_mask: Optional[Tensor] = None, # mask used for self-attention
                cross_attn_mask: Optional[Tensor] = None, # mask used for cross-attention
            ):
        # self attention
        if self.self_attn is not None:
            # import ipdb; ipdb.set_trace()
            if self.decoder_sa_type == 'sa':
                q = k = self.with_pos_embed(tgt, tgt_query_pos)
                tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
                tgt = tgt + self.dropout2(tgt2)
                tgt = self.norm2(tgt)
            elif self.decoder_sa_type == 'ca_label':
                # import ipdb; ipdb.set_trace()
                # q = self.with_pos_embed(tgt, tgt_query_pos)
                bs = tgt.shape[1]
                k = v = self.label_embedding.weight[:, None, :].repeat(1, bs, 1)
                tgt2 = self.self_attn(tgt, k, v, attn_mask=self_attn_mask)[0]
                tgt = tgt + self.dropout2(tgt2)
                tgt = self.norm2(tgt)
            elif self.decoder_sa_type == 'ca_content':
                tgt2 = self.self_attn(self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
                            tgt_reference_points.transpose(0, 1).contiguous(),
                            memory.transpose(0, 1), memory_spatial_shapes, memory_level_start_index, memory_key_padding_mask).transpose(0, 1)
                tgt = tgt + self.dropout2(tgt2)
                tgt = self.norm2(tgt)
            else:
                raise NotImplementedError("Unknown decoder_sa_type {}".format(self.decoder_sa_type))

        return tgt

    def forward_ca(self,
                # for tgt
                tgt: Optional[Tensor],  # nq, bs, d_model
                tgt_query_pos: Optional[Tensor] = None, # pos for query. MLP(Sine(pos))
                tgt_query_sine_embed: Optional[Tensor] = None, # pos for query. Sine(pos)
                tgt_key_padding_mask: Optional[Tensor] = None,
                tgt_reference_points: Optional[Tensor] = None, # nq, bs, 4

                # for memory
                memory: Optional[Tensor] = None, # hw, bs, d_model
                memory_key_padding_mask: Optional[Tensor] = None,
                memory_level_start_index: Optional[Tensor] = None, # num_levels
                memory_spatial_shapes: Optional[Tensor] = None, # bs, num_levels, 2
                memory_pos: Optional[Tensor] = None, # pos for memory

                # sa
                self_attn_mask: Optional[Tensor] = None, # mask used for self-attention
                cross_attn_mask: Optional[Tensor] = None, # mask used for cross-attention
            ):
        # cross attention
        # import ipdb; ipdb.set_trace()
        if self.key_aware_type is not None:

            if self.key_aware_type == 'mean':
                tgt = tgt + memory.mean(0, keepdim=True)
            elif self.key_aware_type == 'proj_mean':
                tgt = tgt + self.key_aware_proj(memory).mean(0, keepdim=True)
            else:
                raise NotImplementedError("Unknown key_aware_type: {}".format(self.key_aware_type))
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
                               tgt_reference_points.transpose(0, 1).contiguous(),
                               memory.transpose(0, 1), memory_spatial_shapes, memory_level_start_index, memory_key_padding_mask).transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        return tgt

    def forward(self,
                # for tgt
                tgt: Optional[Tensor],  # nq, bs, d_model
                tgt_query_pos: Optional[Tensor] = None, # pos for query. MLP(Sine(pos))
                tgt_query_sine_embed: Optional[Tensor] = None, # pos for query. Sine(pos)
                tgt_key_padding_mask: Optional[Tensor] = None,
                tgt_reference_points: Optional[Tensor] = None, # nq, bs, 4

                # for memory
                memory: Optional[Tensor] = None, # hw, bs, d_model
                memory_key_padding_mask: Optional[Tensor] = None,
                memory_level_start_index: Optional[Tensor] = None, # num_levels
                memory_spatial_shapes: Optional[Tensor] = None, # bs, num_levels, 2
                memory_pos: Optional[Tensor] = None, # pos for memory

                # sa
                self_attn_mask: Optional[Tensor] = None, # mask used for self-attention
                cross_attn_mask: Optional[Tensor] = None, # mask used for cross-attention
            ):

        for funcname in self.module_seq:
            # if os.environ.get('IPDB_DEBUG_SHILONG') == 'INFO':
            #     import ipdb; ipdb.set_trace()
            if funcname == 'ffn':
                tgt = self.forward_ffn(tgt)
            elif funcname == 'ca':
                tgt = self.forward_ca(tgt, tgt_query_pos, tgt_query_sine_embed, \
                    tgt_key_padding_mask, tgt_reference_points, \
                        memory, memory_key_padding_mask, memory_level_start_index, \
                            memory_spatial_shapes, memory_pos, self_attn_mask, cross_attn_mask)
            elif funcname == 'sa':
                tgt = self.forward_sa(tgt, tgt_query_pos, tgt_query_sine_embed, \
                    tgt_key_padding_mask, tgt_reference_points, \
                        memory, memory_key_padding_mask, memory_level_start_index, \
                            memory_spatial_shapes, memory_pos, self_attn_mask, cross_attn_mask)
            else:
                raise ValueError('unknown funcname {}'.format(funcname))

        return tgt



class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False, use_dab=False, d_model=256, query_dim=4):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        assert return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None
        self.use_dab = use_dab
        self.d_model = d_model
        self.query_dim = query_dim
        if use_dab:
            self.query_scale = MLP(d_model, d_model, d_model, 2)
            self.ref_point_head = MLP(2 * d_model, d_model, d_model, 2)


    def forward(self, tgt, reference_points, src, src_spatial_shapes,
                src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        output = tgt
        if self.use_dab:
            assert query_pos is None

        intermediate = []
        intermediate_reference_points = [reference_points]
        for layer_id, layer in enumerate(self.layers):
            # import ipdb; ipdb.set_trace()
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None] # bs, nq, 4, 4
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]

            if self.use_dab:
                # import ipdb; ipdb.set_trace()
                query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :]) # bs, nq, 256*2
                raw_query_pos = self.ref_point_head(query_sine_embed) # bs, nq, 256
                pos_scale = self.query_scale(output) if layer_id != 0 else 1
                query_pos = pos_scale * raw_query_pos

            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                box_holder = self.bbox_embed(output)
                box_holder[..., :self.query_dim] += inverse_sigmoid(reference_points)
                new_reference_points = box_holder[..., :self.query_dim].sigmoid()
                reference_points = new_reference_points.detach()
                if layer_id != self.num_layers - 1:
                    intermediate_reference_points.append(new_reference_points)

            intermediate.append(output)

        return torch.stack(intermediate), torch.stack(intermediate_reference_points)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.ddetr_num_feature_levels,
        dec_n_points=args.ddetr_dec_n_points,
        enc_n_points=args.ddetr_enc_n_points,
        two_stage=args.ddetr_two_stage,
        two_stage_num_proposals=args.num_queries,
        use_dab=args.ddetr_use_dab,
        high_dim_query_update=args.ddetr_high_dim_query_update,
        no_sine_embed=args.ddetr_no_sine_embed)
