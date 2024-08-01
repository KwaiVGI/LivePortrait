_base_ = ['coco_transformer.py']

use_label_enc = True

num_classes=2

lr = 0.0001
param_dict_type = 'default'
lr_backbone = 1e-05
lr_backbone_names = ['backbone.0']
lr_linear_proj_names = ['reference_points', 'sampling_offsets']
lr_linear_proj_mult = 0.1
ddetr_lr_param = False
batch_size = 2
weight_decay = 0.0001
epochs = 12
lr_drop = 11
save_checkpoint_interval = 100
clip_max_norm = 0.1
onecyclelr = False
multi_step_lr = False
lr_drop_list = [33, 45]


modelname = 'UniPose'
frozen_weights = None
backbone = 'swin_T_224_1k'


dilation = False
position_embedding = 'sine'
pe_temperatureH = 20
pe_temperatureW = 20
return_interm_indices = [1, 2, 3]
backbone_freeze_keywords = None
enc_layers = 6
dec_layers = 6
unic_layers = 0
pre_norm = False
dim_feedforward = 2048
hidden_dim = 256
dropout = 0.0
nheads = 8
num_queries = 900
query_dim = 4
num_patterns = 0
pdetr3_bbox_embed_diff_each_layer = False
pdetr3_refHW = -1
random_refpoints_xy = False
fix_refpoints_hw = -1
dabdetr_yolo_like_anchor_update = False
dabdetr_deformable_encoder = False
dabdetr_deformable_decoder = False
use_deformable_box_attn = False
box_attn_type = 'roi_align'
dec_layer_number = None
num_feature_levels = 4
enc_n_points = 4
dec_n_points = 4
decoder_layer_noise = False
dln_xy_noise = 0.2
dln_hw_noise = 0.2
add_channel_attention = False
add_pos_value = False
two_stage_type = 'standard'
two_stage_pat_embed = 0
two_stage_add_query_num = 0
two_stage_bbox_embed_share = False
two_stage_class_embed_share = False
two_stage_learn_wh = False
two_stage_default_hw = 0.05
two_stage_keep_all_tokens = False
num_select = 50
transformer_activation = 'relu'
batch_norm_type = 'FrozenBatchNorm2d'
masks = False

decoder_sa_type = 'sa' # ['sa', 'ca_label', 'ca_content']
matcher_type = 'HungarianMatcher' # or SimpleMinsumMatcher
decoder_module_seq = ['sa', 'ca', 'ffn']
nms_iou_threshold = -1

dec_pred_bbox_embed_share = True
dec_pred_class_embed_share = True


use_dn = True
dn_number = 100
dn_box_noise_scale = 1.0
dn_label_noise_ratio = 0.5
dn_label_coef=1.0
dn_bbox_coef=1.0
embed_init_tgt = True
dn_labelbook_size = 2000

match_unstable_error = True

# for ema
use_ema = True
ema_decay = 0.9997
ema_epoch = 0

use_detached_boxes_dec_out = False

max_text_len = 256
shuffle_type = None

use_text_enhancer = True
use_fusion_layer = True

use_checkpoint = False # True
use_transformer_ckpt = True
text_encoder_type = 'bert-base-uncased'

use_text_cross_attention = True
text_dropout = 0.0
fusion_dropout = 0.0
fusion_droppath = 0.1

num_body_points=68
binary_query_selection = False
use_cdn = True
ffn_extra_layernorm = False

fix_size=False
