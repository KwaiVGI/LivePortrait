import torch
from src.utils.helper import load_model

model_config =  {'model_params': {'appearance_feature_extractor_params': {'image_channel': 3, 'block_expansion': 64, 'num_down_blocks': 2, 'max_features': 512, 'reshape_channel': 32, 'reshape_depth': 16, 'num_resblocks': 6}, 'motion_extractor_params': {'num_kp': 21, 'backbone': 'convnextv2_tiny'}, 'warping_module_params': {'num_kp': 21, 'block_expansion': 64, 'max_features': 512, 'num_down_blocks': 2, 'reshape_channel': 32, 'estimate_occlusion_map': True, 'dense_motion_params': {'block_expansion': 32, 'max_features': 1024, 'num_blocks': 5, 'reshape_depth': 16, 'compress': 4}}, 'spade_generator_params': {'upscale': 2, 'block_expansion': 64, 'max_features': 512, 'num_down_blocks': 2}, 'stitching_retargeting_module_params': {'stitching': {'input_size': 126, 'hidden_sizes': [128, 128, 64], 'output_size': 65}, 'lip': {'input_size': 65, 'hidden_sizes': [128, 128, 64], 'output_size': 63}, 'eye': {'input_size': 66, 'hidden_sizes': [256, 256, 128, 128, 64], 'output_size': 63}}}},


def trace_appearance_feature_extractor():
    appearance_feature_extractor = load_model(
        ckpt_path="/mnt/x/1_projects/relight/LivePortrait/src/config/../../pretrained_weights/liveportrait/base_models/appearance_feature_extractor.pth",
        model_config=model_config,
        device=0,
        model_type='appearance_feature_extractor')

    with torch.no_grad():
        appearance_feature_extractor.eval()
        appearance_feature_extractor = torch.jit.script(appearance_feature_extractor)

    torch.jit.save(appearance_feature_extractor, "build/appearance_feature_extractor.pt")

# def trace_appearance_feature_extractor():


def trace_motion_extractor():
    motion_extractor = load_model(
        ckpt_path="/mnt/x/1_projects/relight/LivePortrait/src/config/../../pretrained_weights/liveportrait/base_models/motion_extractor.pth",
        model_config=model_config,
        device=0,
        model_type='motion_extractor')

    with torch.no_grad():
        motion_extractor.eval()
        motion_extractor = torch.jit.script(motion_extractor)

    # torch.jit.save(motion_extractor, "build/motion_extractor.pt")

trace_motion_extractor()


def trace_warping_module():
    warping_module = load_model(
        ckpt_path="/mnt/x/1_projects/relight/LivePortrait/src/config/../../pretrained_weights/liveportrait/base_models/warping_module.pth",
        model_config=model_config,
        device=0,
        model_type='warping_module')

    with torch.no_grad():
        warping_module.eval()
        warping_module = torch.jit.script(warping_module)

    torch.jit.save(warping_module, "build/warping_module.pt")

# def trace_warping_module():

