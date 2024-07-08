from cog import BasePredictor, Input, Path, File

from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.live_portrait_pipeline import LivePortraitPipeline
import requests

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.live_portrait_pipeline = LivePortraitPipeline(
            inference_cfg=InferenceConfig(),
            crop_cfg=CropConfig()
        )

    def predict(
        self,
        input_image_path: Path = Input(description="Portrait image"),
        input_video_path: Path = Input(description="Driving video"),
        flag_relative_input: bool = Input(description="relative motion", default=True),
        flag_do_crop_input: bool = Input(description="We recommend checking the do crop option when facial areas occupy a relatively small portion of your image.", default=True),
        flag_pasteback: bool = Input(description="paste-back", default=True),
    ) -> Path:
        """Run a single prediction on the model"""

        user_args = ArgumentConfig(
            flag_relative=flag_relative_input,
            flag_do_crop=flag_do_crop_input,
            flag_pasteback=flag_pasteback,
            source_image=input_image_path,
            driving_info=str(input_video_path),
            output_dir="/tmp/"
        )
        self.live_portrait_pipeline.cropper.update_config(user_args.__dict__)
        self.live_portrait_pipeline.live_portrait_wrapper.update_config(user_args.__dict__)

        video_path, _ = self.live_portrait_pipeline.execute(
            user_args
        )

        return Path(video_path)
