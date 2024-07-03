# coding: utf-8

"""
[WIP] Pipeline for video template preparation
"""

import tyro
from src.config.crop_config import CropConfig
from src.config.inference_config import InferenceConfig
from src.config.argument_config import ArgumentConfig
from src.template_maker import TemplateMaker


def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})


def main():
    # set tyro theme
    tyro.extras.set_accent_color("bright_cyan")
    args = tyro.cli(ArgumentConfig)

    # specify configs for inference
    inference_cfg = partial_fields(InferenceConfig, args.__dict__)  # use attribute of args to initial InferenceConfig
    crop_cfg = partial_fields(CropConfig, args.__dict__)  # use attribute of args to initial CropConfig

    video_template_maker = TemplateMaker(
        inference_cfg=inference_cfg,
        crop_cfg=crop_cfg
    )

    # run
    video_template_maker.make_motion_template(args.driving_video_path, args.template_output_dir)


if __name__ == '__main__':
    main()
