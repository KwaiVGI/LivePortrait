# coding: utf-8

"""
Benchmark the inference speed of each module in LivePortrait.

TODO: heavy GPT style, need to refactor
"""

import yaml
import torch
import time
import numpy as np
from src.utils.helper import load_model, concat_feat
from src.config.inference_config import InferenceConfig


def initialize_inputs(batch_size=1):
    """
    Generate random input tensors and move them to GPU
    """
    feature_3d = torch.randn(batch_size, 32, 16, 64, 64).cuda().half()
    kp_source = torch.randn(batch_size, 21, 3).cuda().half()
    kp_driving = torch.randn(batch_size, 21, 3).cuda().half()
    source_image = torch.randn(batch_size, 3, 256, 256).cuda().half()
    generator_input = torch.randn(batch_size, 256, 64, 64).cuda().half()
    eye_close_ratio = torch.randn(batch_size, 3).cuda().half()
    lip_close_ratio = torch.randn(batch_size, 2).cuda().half()
    feat_stitching = concat_feat(kp_source, kp_driving).half()
    feat_eye = concat_feat(kp_source, eye_close_ratio).half()
    feat_lip = concat_feat(kp_source, lip_close_ratio).half()

    inputs = {
        'feature_3d': feature_3d,
        'kp_source': kp_source,
        'kp_driving': kp_driving,
        'source_image': source_image,
        'generator_input': generator_input,
        'feat_stitching': feat_stitching,
        'feat_eye': feat_eye,
        'feat_lip': feat_lip
    }

    return inputs


def load_and_compile_models(cfg, model_config):
    """
    Load and compile models for inference
    """
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True  # Suppress errors and fall back to eager execution

    appearance_feature_extractor = load_model(cfg.checkpoint_F, model_config, cfg.device_id, 'appearance_feature_extractor')
    motion_extractor = load_model(cfg.checkpoint_M, model_config, cfg.device_id, 'motion_extractor')
    warping_module = load_model(cfg.checkpoint_W, model_config, cfg.device_id, 'warping_module')
    spade_generator = load_model(cfg.checkpoint_G, model_config, cfg.device_id, 'spade_generator')
    stitching_retargeting_module = load_model(cfg.checkpoint_S, model_config, cfg.device_id, 'stitching_retargeting_module')

    models_with_params = [
        ('Appearance Feature Extractor', appearance_feature_extractor),
        ('Motion Extractor', motion_extractor),
        ('Warping Network', warping_module),
        ('SPADE Decoder', spade_generator)
    ]

    compiled_models = {}
    for name, model in models_with_params:
        model = model.half()
        model = torch.compile(model, mode='max-autotune')  # Optimize for inference
        model.eval()  # Switch to evaluation mode
        compiled_models[name] = model

    retargeting_models = ['stitching', 'eye', 'lip']
    for retarget in retargeting_models:
        module = stitching_retargeting_module[retarget].half()
        module = torch.compile(module, mode='max-autotune')  # Optimize for inference
        module.eval()  # Switch to evaluation mode
        stitching_retargeting_module[retarget] = module

    return compiled_models, stitching_retargeting_module


def warm_up_models(compiled_models, stitching_retargeting_module, inputs):
    """
    Warm up models to prepare them for benchmarking
    """
    print("Warm up start!")
    with torch.no_grad():
        for _ in range(10):
            compiled_models['Appearance Feature Extractor'](inputs['source_image'])
            compiled_models['Motion Extractor'](inputs['source_image'])
            compiled_models['Warping Network'](inputs['feature_3d'], inputs['kp_driving'], inputs['kp_source'])
            compiled_models['SPADE Decoder'](inputs['generator_input'])  # Adjust input as required
            stitching_retargeting_module['stitching'](inputs['feat_stitching'])
            stitching_retargeting_module['eye'](inputs['feat_eye'])
            stitching_retargeting_module['lip'](inputs['feat_lip'])
    print("Warm up end!")


def measure_inference_times(compiled_models, stitching_retargeting_module, inputs):
    """
    Measure inference times for each model
    """
    times = {name: [] for name in compiled_models.keys()}
    times['Retargeting Models'] = []

    overall_times = []

    with torch.no_grad():
        for _ in range(100):
            torch.cuda.synchronize()
            overall_start = time.time()

            start = time.time()
            compiled_models['Appearance Feature Extractor'](inputs['source_image'])
            torch.cuda.synchronize()
            times['Appearance Feature Extractor'].append(time.time() - start)

            start = time.time()
            compiled_models['Motion Extractor'](inputs['source_image'])
            torch.cuda.synchronize()
            times['Motion Extractor'].append(time.time() - start)

            start = time.time()
            compiled_models['Warping Network'](inputs['feature_3d'], inputs['kp_driving'], inputs['kp_source'])
            torch.cuda.synchronize()
            times['Warping Network'].append(time.time() - start)

            start = time.time()
            compiled_models['SPADE Decoder'](inputs['generator_input'])  # Adjust input as required
            torch.cuda.synchronize()
            times['SPADE Decoder'].append(time.time() - start)

            start = time.time()
            stitching_retargeting_module['stitching'](inputs['feat_stitching'])
            stitching_retargeting_module['eye'](inputs['feat_eye'])
            stitching_retargeting_module['lip'](inputs['feat_lip'])
            torch.cuda.synchronize()
            times['Retargeting Models'].append(time.time() - start)

            overall_times.append(time.time() - overall_start)

    return times, overall_times


def print_benchmark_results(compiled_models, stitching_retargeting_module, retargeting_models, times, overall_times):
    """
    Print benchmark results with average and standard deviation of inference times
    """
    average_times = {name: np.mean(times[name]) * 1000 for name in times.keys()}
    std_times = {name: np.std(times[name]) * 1000 for name in times.keys()}

    for name, model in compiled_models.items():
        num_params = sum(p.numel() for p in model.parameters())
        num_params_in_millions = num_params / 1e6
        print(f"Number of parameters for {name}: {num_params_in_millions:.2f} M")

    for index, retarget in enumerate(retargeting_models):
        num_params = sum(p.numel() for p in stitching_retargeting_module[retarget].parameters())
        num_params_in_millions = num_params / 1e6
        print(f"Number of parameters for part_{index} in Stitching and Retargeting Modules: {num_params_in_millions:.2f} M")

    for name, avg_time in average_times.items():
        std_time = std_times[name]
        print(f"Average inference time for {name} over 100 runs: {avg_time:.2f} ms (std: {std_time:.2f} ms)")


def main():
    """
    Main function to benchmark speed and model parameters
    """
    # Sample input tensors
    inputs = initialize_inputs()

    # Load configuration
    cfg = InferenceConfig(device_id=0)
    model_config_path = cfg.models_config
    with open(model_config_path, 'r') as file:
        model_config = yaml.safe_load(file)

    # Load and compile models
    compiled_models, stitching_retargeting_module = load_and_compile_models(cfg, model_config)

    # Warm up models
    warm_up_models(compiled_models, stitching_retargeting_module, inputs)

    # Measure inference times
    times, overall_times = measure_inference_times(compiled_models, stitching_retargeting_module, inputs)

    # Print benchmark results
    print_benchmark_results(compiled_models, stitching_retargeting_module, ['stitching', 'eye', 'lip'], times, overall_times)


if __name__ == "__main__":
    main()
