# coding: utf-8

import os.path as osp
import tyro
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.live_portrait_pipeline import LivePortraitPipeline
import subprocess


def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})


def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False


def fast_check_args(args: ArgumentConfig):
    if not osp.exists(args.source_image):
        raise FileNotFoundError(f"source image not found: {args.source_image}")
    if not osp.exists(args.driving_info):
        raise FileNotFoundError(f"driving info not found: {args.driving_info}")


def main():
    # set tyro theme
    tyro.extras.set_accent_color("bright_cyan")
    args = tyro.cli(ArgumentConfig)

    if not fast_check_ffmpeg():
        raise ImportError(
            "FFmpeg is not installed. Please install FFmpeg before running this script. https://ffmpeg.org/download.html"
        )
    # fast check the args
    fast_check_args(args)

    # specify configs for inference
    inference_cfg = partial_fields(
        InferenceConfig, args.__dict__
    )  # use attribute of args to initial InferenceConfig
    crop_cfg = partial_fields(
        CropConfig, args.__dict__
    )  # use attribute of args to initial CropConfig

    live_portrait_pipeline = LivePortraitPipeline(
        inference_cfg=inference_cfg, crop_cfg=crop_cfg
    )

    # run
    live_portrait_pipeline.execute(args)


if __name__ == "__main__":
    main()
