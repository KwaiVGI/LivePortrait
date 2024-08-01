import torch


def prepare_for_mask(kpt_mask):


    tgt_size2 = 50 * 69
    attn_mask2 = torch.ones(kpt_mask.shape[0], 8, tgt_size2, tgt_size2).to('cuda') < 0
    group_bbox_kpt = 69
    num_group=50
    for matchj in range(num_group * group_bbox_kpt):
        sj = (matchj // group_bbox_kpt) * group_bbox_kpt
        ej = (matchj // group_bbox_kpt + 1)*group_bbox_kpt
        if sj > 0:
            attn_mask2[:,:,matchj, :sj] = True
        if ej < num_group * group_bbox_kpt:
            attn_mask2[:,:,matchj, ej:] = True


    bs, length = kpt_mask.shape
    equal_mask = kpt_mask[:, :, None] == kpt_mask[:, None, :]
    equal_mask= equal_mask.unsqueeze(1).repeat(1,8,1,1)
    for idx in range(num_group):
        start_idx = idx * length
        end_idx = (idx + 1) * length
        attn_mask2[:, :,start_idx:end_idx, start_idx:end_idx][equal_mask] = False
        attn_mask2[:, :,start_idx:end_idx, start_idx:end_idx][~equal_mask] = True




    input_query_label = None
    input_query_bbox = None
    attn_mask = None
    dn_meta = None

    return input_query_label, input_query_bbox, attn_mask, attn_mask2.flatten(0,1), dn_meta


def post_process(outputs_class, outputs_coord, dn_meta, aux_loss, _set_aux_loss):

    if dn_meta and dn_meta['pad_size'] > 0:

        output_known_class = [outputs_class_i[:, :dn_meta['pad_size'], :] for outputs_class_i in outputs_class]
        output_known_coord = [outputs_coord_i[:, :dn_meta['pad_size'], :] for outputs_coord_i in outputs_coord]

        outputs_class = [outputs_class_i[:, dn_meta['pad_size']:, :] for outputs_class_i in outputs_class]
        outputs_coord = [outputs_coord_i[:, dn_meta['pad_size']:, :] for outputs_coord_i in outputs_coord]

        out = {'pred_logits': output_known_class[-1], 'pred_boxes': output_known_coord[-1]}
        if aux_loss:
            out['aux_outputs'] = _set_aux_loss(output_known_class, output_known_coord)
        dn_meta['output_known_lbs_bboxes'] = out
    return outputs_class, outputs_coord


