# ------------------------------------------------------------------------
# ED-Pose
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
import copy
import math
import os
from typing import List
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops.boxes import nms
from util.keypoint_ops import keypoint_xyzxyz_to_xyxyzz
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from .backbone import build_backbone
from .deformable_transformer import build_deformable_transformer
from .utils import sigmoid_focal_loss, MLP

from ..registry import MODULE_BUILD_FUNCS
from .mask_generate import prepare_for_mask, post_process
import random
from .utils import sigmoid_focal_loss, MLP, _get_activation_fn, gen_sineembed_for_position
from pathlib import Path


class UniPose(nn.Module):
    """ This is the Cross-Attention Detector module that performs object detection """

    def __init__(self, backbone, transformer, num_classes, num_queries,
                 aux_loss=False, iter_update=False,
                 query_dim=2,
                 random_refpoints_xy=False,
                 fix_refpoints_hw=-1,
                 num_feature_levels=1,
                 nheads=8,
                 # two stage
                 two_stage_type='no',  # ['no', 'standard']
                 two_stage_add_query_num=0,
                 dec_pred_class_embed_share=True,
                 dec_pred_bbox_embed_share=True,
                 two_stage_class_embed_share=True,
                 two_stage_bbox_embed_share=True,
                 decoder_sa_type='sa',
                 num_patterns=0,
                 dn_number=100,
                 dn_box_noise_scale=0.4,
                 dn_label_noise_ratio=0.5,
                 dn_labelbook_size=100,
                 use_label_enc=True,

                 text_encoder_type='bert-base-uncased',

                 binary_query_selection=False,
                 use_cdn=True,
                 sub_sentence_present=True,
                 num_body_points=68,
                 num_box_decoder_layers=2,
                 ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.

            fix_refpoints_hw: -1(default): learn w and h for each box seperately
                                >0 : given fixed number
                                -2 : learn a shared w and h
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.use_label_enc = use_label_enc
        if use_label_enc:
            self.label_enc = nn.Embedding(dn_labelbook_size + 1, hidden_dim)
        else:
            raise NotImplementedError
            self.label_enc = None
        self.max_text_len = 256
        self.binary_query_selection = binary_query_selection
        self.sub_sentence_present = sub_sentence_present

        # setting query dim
        self.query_dim = query_dim
        assert query_dim == 4
        self.random_refpoints_xy = random_refpoints_xy
        self.fix_refpoints_hw = fix_refpoints_hw

        # for dn training
        self.num_patterns = num_patterns
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size
        self.use_cdn = use_cdn


        self.projection = MLP(512, hidden_dim, hidden_dim, 3)

        self.projection_kpt = MLP(512, hidden_dim, hidden_dim, 3)


        device = "cuda" if torch.cuda.is_available() else "cpu"
        # model, _ = clip.load("ViT-B/32", device=device)
        # self.clip_model = model
        # visual_parameters = list(self.clip_model.visual.parameters())
        # #
        # for param in visual_parameters:
        #     param.requires_grad = False

        self.pos_proj = nn.Linear(hidden_dim, 768)
        self.padding = nn.Embedding(1, 768)

        # prepare input projection layers
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            assert two_stage_type == 'no', "two_stage_type should be no if num_feature_levels=1 !!!"
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.box_pred_damping = box_pred_damping = None

        self.iter_update = iter_update
        assert iter_update, "Why not iter_update?"

        # prepare pred layers
        self.dec_pred_class_embed_share = dec_pred_class_embed_share
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        # prepare class & box embed
        _class_embed = ContrastiveAssign()



        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        _pose_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        _pose_hw_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        nn.init.constant_(_pose_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_pose_embed.layers[-1].bias.data, 0)

        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for i in range(transformer.num_decoder_layers)]
        else:
            box_embed_layerlist = [copy.deepcopy(_bbox_embed) for i in range(transformer.num_decoder_layers)]
        if dec_pred_class_embed_share:
            class_embed_layerlist = [_class_embed for i in range(transformer.num_decoder_layers)]
        else:
            class_embed_layerlist = [copy.deepcopy(_class_embed) for i in range(transformer.num_decoder_layers)]


        if dec_pred_bbox_embed_share:

            pose_embed_layerlist = [_pose_embed for i in
                                    range(transformer.num_decoder_layers - num_box_decoder_layers + 1)]
        else:
            pose_embed_layerlist = [copy.deepcopy(_pose_embed) for i in
                                    range(transformer.num_decoder_layers - num_box_decoder_layers + 1)]

        pose_hw_embed_layerlist = [_pose_hw_embed for i in
                                   range(transformer.num_decoder_layers - num_box_decoder_layers)]


        self.num_box_decoder_layers = num_box_decoder_layers
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.num_body_points = num_body_points
        self.pose_embed = nn.ModuleList(pose_embed_layerlist)
        self.pose_hw_embed = nn.ModuleList(pose_hw_embed_layerlist)

        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        self.transformer.decoder.pose_embed = self.pose_embed
        self.transformer.decoder.pose_hw_embed = self.pose_hw_embed

        self.transformer.decoder.num_body_points = num_body_points


        # two stage
        self.two_stage_type = two_stage_type
        self.two_stage_add_query_num = two_stage_add_query_num
        assert two_stage_type in ['no', 'standard'], "unknown param {} of two_stage_type".format(two_stage_type)
        if two_stage_type != 'no':
            if two_stage_bbox_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)

            if two_stage_class_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

            self.refpoint_embed = None
            if self.two_stage_add_query_num > 0:
                self.init_ref_points(two_stage_add_query_num)

        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ['sa', 'ca_label', 'ca_content']
        # self.replace_sa_with_double_ca = replace_sa_with_double_ca
        if decoder_sa_type == 'ca_label':
            self.label_embedding = nn.Embedding(num_classes, hidden_dim)
            for layer in self.transformer.decoder.layers:
                layer.label_embedding = self.label_embedding
        else:
            for layer in self.transformer.decoder.layers:
                layer.label_embedding = None
            self.label_embedding = None

        self._reset_parameters()

    def open_set_transfer_init(self):
        for name, param in self.named_parameters():
            if 'fusion_layers' in name:
                continue
            if 'ca_text' in name:
                continue
            if 'catext_norm' in name:
                continue
            if 'catext_dropout' in name:
                continue
            if "text_layers" in name:
                continue
            if 'bert' in name:
                continue
            if 'bbox_embed' in name:
                continue
            if 'label_enc.weight' in name:
                continue
            if 'feat_map' in name:
                continue
            if 'enc_output' in name:
                continue

            param.requires_grad_(False)

        # import ipdb; ipdb.set_trace()

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)

        if self.random_refpoints_xy:
            # import ipdb; ipdb.set_trace()
            self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False

        if self.fix_refpoints_hw > 0:
            print("fix_refpoints_hw: {}".format(self.fix_refpoints_hw))
            assert self.random_refpoints_xy
            self.refpoint_embed.weight.data[:, 2:] = self.fix_refpoints_hw
            self.refpoint_embed.weight.data[:, 2:] = inverse_sigmoid(self.refpoint_embed.weight.data[:, 2:])
            self.refpoint_embed.weight.data[:, 2:].requires_grad = False
        elif int(self.fix_refpoints_hw) == -1:
            pass
        elif int(self.fix_refpoints_hw) == -2:
            print('learn a shared h and w')
            assert self.random_refpoints_xy
            self.refpoint_embed = nn.Embedding(use_num_queries, 2)
            self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False
            self.hw_embed = nn.Embedding(1, 1)
        else:
            raise NotImplementedError('Unknown fix_refpoints_hw {}'.format(self.fix_refpoints_hw))

    def forward(self, samples: NestedTensor, targets: List = None, **kw):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """

        captions = [t['instance_text_prompt'] for t in targets]
        bs=len(captions)
        tensor_list = [tgt["object_embeddings_text"] for tgt in targets]
        max_size = 350
        padded_tensors = [torch.cat([tensor, torch.zeros(max_size - tensor.size(0), tensor.size(1),device=tensor.device)]) if tensor.size(0) < max_size else tensor for tensor in tensor_list]
        object_embeddings_text = torch.stack(padded_tensors)

        kpts_embeddings_text = torch.stack([tgt["kpts_embeddings_text"] for tgt in targets])[:, :self.num_body_points]
        encoded_text=self.projection(object_embeddings_text) # bs, 81, 101, 256
        kpt_embeddings_specific=self.projection_kpt(kpts_embeddings_text) # bs, 81, 101, 256


        kpt_vis = torch.stack([tgt["kpt_vis_text"] for tgt in targets])[:, :self.num_body_points]
        kpt_mask = torch.cat((torch.ones_like(kpt_vis, device=kpt_vis.device)[..., 0].unsqueeze(-1), kpt_vis), dim=-1)


        num_classes = encoded_text.shape[1] # bs, 81, 101, 256
        text_self_attention_masks = torch.eye(num_classes).unsqueeze(0).expand(bs, -1, -1).bool().to(samples.device)
        text_token_mask = torch.zeros(samples.shape[0],num_classes).to(samples.device)>0
        for i in range(bs):
            text_token_mask[i,:len(captions[i])]=True

        position_ids = torch.zeros(samples.shape[0], num_classes).to(samples.device)

        for i in range(bs):
            position_ids[i,:len(captions[i])]= 1


        text_dict = {
            'encoded_text': encoded_text, # bs, 195, d_model
            'text_token_mask': text_token_mask, # bs, 195
            'position_ids': position_ids, # bs, 195
            'text_self_attention_masks': text_self_attention_masks # bs, 195,195
        }


        # import ipdb; ipdb.set_trace()

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, poss = self.backbone(samples)
        if os.environ.get("SHILONG_AMP_INFNAN_DEBUG") == '1':
            import ipdb;
            ipdb.set_trace()


        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)

        if self.label_enc is not None:
            label_enc = self.label_enc
        else:
            raise NotImplementedError
            label_enc = encoded_text
        if self.dn_number > 0 or targets is not None:
            input_query_label, input_query_bbox, attn_mask, attn_mask2, dn_meta = \
                prepare_for_mask(kpt_mask=kpt_mask)
        else:
            assert targets is None
            input_query_bbox = input_query_label = attn_mask = attn_mask2 = dn_meta = None


        hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(srcs, masks, input_query_bbox, poss,
                                                                                 input_query_label, attn_mask, attn_mask2,
                                                                                 text_dict, dn_meta,targets,kpt_embeddings_specific)

        # In case num object=0
        if self.label_enc is not None:
            hs[0] += self.label_enc.weight[0, 0] * 0.0

        hs[0] += self.pos_proj.weight[0, 0] * 0.0
        hs[0] += self.pos_proj.bias[0] * 0.0
        hs[0] += self.padding.weight[0, 0] * 0.0

        num_group = 50
        effective_dn_number = dn_meta['pad_size'] if self.training else 0
        outputs_coord_list = []
        outputs_class = []


        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_cls_embed, layer_hs) in enumerate(
                zip(reference[:-1], self.bbox_embed, self.class_embed, hs)):


            if dec_lid < self.num_box_decoder_layers:
                layer_delta_unsig = layer_bbox_embed(layer_hs)
                layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
                layer_outputs_unsig = layer_outputs_unsig.sigmoid()
                layer_cls = layer_cls_embed(layer_hs, text_dict)
                outputs_coord_list.append(layer_outputs_unsig)
                outputs_class.append(layer_cls)


            else:

                layer_hs_bbox_dn = layer_hs[:, :effective_dn_number, :]
                layer_hs_bbox_norm = layer_hs[:, effective_dn_number:, :][:, 0::(self.num_body_points + 1), :]
                bs = layer_ref_sig.shape[0]
                reference_before_sigmoid_bbox_dn = layer_ref_sig[:, :effective_dn_number, :]
                reference_before_sigmoid_bbox_norm = layer_ref_sig[:, effective_dn_number:, :][:,
                                                     0::(self.num_body_points + 1), :]
                layer_delta_unsig_dn = layer_bbox_embed(layer_hs_bbox_dn)
                layer_delta_unsig_norm = layer_bbox_embed(layer_hs_bbox_norm)
                layer_outputs_unsig_dn = layer_delta_unsig_dn + inverse_sigmoid(reference_before_sigmoid_bbox_dn)
                layer_outputs_unsig_dn = layer_outputs_unsig_dn.sigmoid()
                layer_outputs_unsig_norm = layer_delta_unsig_norm + inverse_sigmoid(reference_before_sigmoid_bbox_norm)
                layer_outputs_unsig_norm = layer_outputs_unsig_norm.sigmoid()
                layer_outputs_unsig = torch.cat((layer_outputs_unsig_dn, layer_outputs_unsig_norm), dim=1)
                layer_cls_dn = layer_cls_embed(layer_hs_bbox_dn, text_dict)
                layer_cls_norm = layer_cls_embed(layer_hs_bbox_norm, text_dict)
                layer_cls = torch.cat((layer_cls_dn, layer_cls_norm), dim=1)
                outputs_class.append(layer_cls)
                outputs_coord_list.append(layer_outputs_unsig)

        # update keypoints
        outputs_keypoints_list = []
        outputs_keypoints_hw = []
        kpt_index = [x for x in range(num_group * (self.num_body_points + 1)) if x % (self.num_body_points + 1) != 0]
        for dec_lid, (layer_ref_sig, layer_hs) in enumerate(zip(reference[:-1], hs)):
            if dec_lid < self.num_box_decoder_layers:
                assert isinstance(layer_hs, torch.Tensor)
                bs = layer_hs.shape[0]
                layer_res = layer_hs.new_zeros((bs, self.num_queries, self.num_body_points * 3))
                outputs_keypoints_list.append(layer_res)
            else:
                bs = layer_ref_sig.shape[0]
                layer_hs_kpt = layer_hs[:, effective_dn_number:, :].index_select(1, torch.tensor(kpt_index,
                                                                                                 device=layer_hs.device))
                delta_xy_unsig = self.pose_embed[dec_lid - self.num_box_decoder_layers](layer_hs_kpt)
                layer_ref_sig_kpt = layer_ref_sig[:, effective_dn_number:, :].index_select(1, torch.tensor(kpt_index,
                                                                                                           device=layer_hs.device))
                layer_outputs_unsig_keypoints = delta_xy_unsig + inverse_sigmoid(layer_ref_sig_kpt[..., :2])
                vis_xy_unsig = torch.ones_like(layer_outputs_unsig_keypoints,
                                               device=layer_outputs_unsig_keypoints.device)
                xyv = torch.cat((layer_outputs_unsig_keypoints, vis_xy_unsig[:, :, 0].unsqueeze(-1)), dim=-1)
                xyv = xyv.sigmoid()
                layer_res = xyv.reshape((bs, num_group, self.num_body_points, 3)).flatten(2, 3)
                layer_hw = layer_ref_sig_kpt[..., 2:].reshape(bs, num_group, self.num_body_points, 2).flatten(2, 3)
                layer_res = keypoint_xyzxyz_to_xyxyzz(layer_res)
                outputs_keypoints_list.append(layer_res)
                outputs_keypoints_hw.append(layer_hw)


        if self.dn_number > 0 and dn_meta is not None:
            outputs_class, outputs_coord_list = \
                post_process(outputs_class, outputs_coord_list,
                                dn_meta, self.aux_loss, self._set_aux_loss)
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord_list[-1],
               'pred_keypoints': outputs_keypoints_list[-1]}

        return out


@MODULE_BUILD_FUNCS.registe_with_name(module_name='UniPose')
def build_unipose(args):

    num_classes = args.num_classes
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deformable_transformer(args)

    try:
        match_unstable_error = args.match_unstable_error
        dn_labelbook_size = args.dn_labelbook_size
    except:
        match_unstable_error = True
        dn_labelbook_size = num_classes

    try:
        dec_pred_class_embed_share = args.dec_pred_class_embed_share
    except:
        dec_pred_class_embed_share = True
    try:
        dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share
    except:
        dec_pred_bbox_embed_share = True

    binary_query_selection = False
    try:
        binary_query_selection = args.binary_query_selection
    except:
        binary_query_selection = False

    use_cdn = True
    try:
        use_cdn = args.use_cdn
    except:
        use_cdn = True

    sub_sentence_present = True
    try:
        sub_sentence_present = args.sub_sentence_present
    except:
        sub_sentence_present = True
    # print('********* sub_sentence_present', sub_sentence_present)

    model = UniPose(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=True,
        iter_update=True,
        query_dim=4,
        random_refpoints_xy=args.random_refpoints_xy,
        fix_refpoints_hw=args.fix_refpoints_hw,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        dec_pred_class_embed_share=dec_pred_class_embed_share,
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
        # two stage
        two_stage_type=args.two_stage_type,
        # box_share
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        decoder_sa_type=args.decoder_sa_type,
        num_patterns=args.num_patterns,
        dn_number=args.dn_number if args.use_dn else 0,
        dn_box_noise_scale=args.dn_box_noise_scale,
        dn_label_noise_ratio=args.dn_label_noise_ratio,
        dn_labelbook_size=dn_labelbook_size,
        use_label_enc=args.use_label_enc,

        text_encoder_type=args.text_encoder_type,

        binary_query_selection=binary_query_selection,
        use_cdn=use_cdn,
        sub_sentence_present=sub_sentence_present
    )

    return model


class ContrastiveAssign(nn.Module):
    def __init__(self, project=False, cal_bias=None, max_text_len=256):
        """
        :param x: query
        :param y: text embed
        :param proj:
        :return:
        """
        super().__init__()
        self.project = project
        self.cal_bias = cal_bias
        self.max_text_len = max_text_len

    def forward(self, x, text_dict):
        """_summary_

        Args:
            x (_type_): _description_
            text_dict (_type_): _description_
            {
                'encoded_text': encoded_text, # bs, 195, d_model
                'text_token_mask': text_token_mask, # bs, 195
                        # True for used tokens. False for padding tokens
            }
        Returns:
            _type_: _description_
        """
        assert isinstance(text_dict, dict)

        y = text_dict['encoded_text']


        max_text_len = y.shape[1]



        text_token_mask = text_dict['text_token_mask']

        if self.cal_bias is not None:
            raise NotImplementedError
            return x @ y.transpose(-1, -2) + self.cal_bias.weight.repeat(x.shape[0], x.shape[1], 1)
        res = x @ y.transpose(-1, -2)
        res.masked_fill_(~text_token_mask[:, None, :], float('-inf'))

        # padding to max_text_len
        new_res = torch.full((*res.shape[:-1], max_text_len), float('-inf'), device=res.device)
        new_res[..., :res.shape[-1]] = res

        return new_res
