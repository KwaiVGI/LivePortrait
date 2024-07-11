# coding: utf-8

import tyro
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.live_portrait_pipeline import LivePortraitPipeline
import subprocess


def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})

def check_ffmpeg_installed():
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except:
        return False

def main():
    # set tyro theme
    tyro.extras.set_accent_color("bright_cyan")
    args = tyro.cli(ArgumentConfig)

    if not check_ffmpeg_installed():
        raise ImportError(
            "FFmpeg is not installed. Please install FFmpeg before running this script. https://ffmpeg.org/download.html"
        )

    # specify configs for inference
    inference_cfg = partial_fields(InferenceConfig, args.__dict__)  # use attribute of args to initial InferenceConfig
    crop_cfg = partial_fields(CropConfig, args.__dict__)  # use attribute of args to initial CropConfig

    live_portrait_pipeline = LivePortraitPipeline(
        inference_cfg=inference_cfg,
        crop_cfg=crop_cfg
    )

    # run
    live_portrait_pipeline.execute(args)


if __name__ == '__main__':
    main()
