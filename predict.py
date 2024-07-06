from cog import BasePredictor, Input, Path

from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.live_portrait_pipeline import LivePortraitPipeline


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.live_portrait_pipeline = LivePortraitPipeline(
            inference_cfg=InferenceConfig(),
            crop_cfg=CropConfig()
        )

    def predict(
        self,
        image: Path = Input(description="Portrait image"),
        driving_info: Path = Input(
            description="driving video or template (.pkl format)"
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        video_path, _ = self.live_portrait_pipeline.execute(
            ArgumentConfig(source_image=image, driving_info=driving_info, output_dir="/tmp/")
        )

        return Path(video_path)
