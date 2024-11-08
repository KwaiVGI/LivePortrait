import argparse
import logging
from fastapi.security import APIKeyHeader
import os
from fastapi import FastAPI, Depends
from typing import Optional, Tuple

from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.live_portrait_pipeline import LivePortraitPipeline
from pydantic import BaseModel, Field, root_validator
from starlette.responses import JSONResponse

import tyro

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NOTE: For now, we use an API key to authenticate the requests.
# This method should only be used for testing purposes.
# For actual deployment, we should use OAuth2 or other more secure methods.
header_scheme = APIKeyHeader(name="X-API-Key")
SERVING_API_KEY = os.getenv("SERVING_API_KEY", "oLjQD5hWDYN5DeAQ4cx5CL3vJYOTXf0c")
OUTPUT_LOCAL_PATH = os.getenv("OUTPUT_LOCAL_PATH", "/tmp/outputs/")

app = FastAPI()

live_portrait_pipeline = None


class LivePortraitRequestProto(BaseModel):
    '''Define the request schema for the Live Portrait API.
    Live Portrait needs two inputs to generate the final output:
    1. The source image/video (src_key or src_local_path)
    2. The driving image/video (driving_key or driving_local_path)
    This server will process source input according to the driving input.
    Some characteristics of the driving input will be transferred and applied to the source input.
    '''
    src_key: Optional[str] = Field(None, description="Object key of the source image/video from COS", min_length=1)
    src_local_path: Optional[str] = Field(
        None, description="Local file path of the source image/video to be processed on the server", min_length=1
    )
    
    driving_key: Optional[str] = Field(None, description="Object key of the driving image/video from COS", min_length=1)
    driving_local_path: Optional[str] = Field(
        None, description="Local file path of the driving image/video to be processed on the server", min_length=1
    )

    @root_validator(pre=True)
    def check_either_image_key_or_local_path(cls, values):
        src_key = values.get("src_key")
        src_local_path = values.get("src_local_path")
        driving_key = values.get("driving_key")
        driving_local_path = values.get("driving_local_path")
        if (src_key and src_local_path) or (not src_key and not src_local_path):
            raise ValueError("Either 'src_key' or 'src_local_path' must be provided, but not both.")
        if (driving_key and driving_local_path) or (not driving_key and not driving_local_path):
            raise ValueError("Either 'driving_key' or 'driving_local_path' must be provided, but not both.")
        return values
    

def init_live_portrait_pipeline():
    # fast_check_args(args)

    # specify configs for inference
    inference_cfg = partial_fields(InferenceConfig, args.__dict__)
    crop_cfg = partial_fields(CropConfig, args.__dict__)

    global live_portrait_pipeline
    live_portrait_pipeline = LivePortraitPipeline(
        inference_cfg=inference_cfg,
        crop_cfg=crop_cfg
    )
    

def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})


def extract_inputs_from_request(request: LivePortraitRequestProto) -> Tuple[str, str]:
    
    src_input = None
    driving_input = None
    
    # 1. Download the image/video from COS to local storage
    if request.src_key:
        storage_dir = "/tmp/sources"
        filename = os.path.basename(request.src_key)
    
    if request.driving_key:
        storage_dir = "/tmp/driving"
        filename = os.path.basename(request.driving_key)
    
    os.makedirs(storage_path, exist_ok=True)  # Ensure the directory exists
    storage_path = os.path.join(storage_dir, filename)
    # Download the image/video from COS
    # download_success = await cos_download_file(request.src_key, src_local_path)
    download_success = True
    if not download_success:
        logger.error(f"Failed to download image: {request.image_key}")
        return (None, None)
    
    # 2. Process the local image/video
    # This step is just for local tests
    if request.src_local_path:
        logger.warning("Using local files is not intended for deployment. This is for testing purposes only.")
        local_image_path = request.local_path
        if not os.path.exists(local_image_path):
            logger.error(f"Local image path does not exist: {local_image_path}")
            return JSONResponse({"message": "Local image path does not exist", "faces_detected": 0}, status_code=500)
    
    if request.driving_local_path:
        logger.warning("Using local files is not intended for deployment. This is for testing purposes only.")
        local_image_path = request.local_path
        if not os.path.exists(local_image_path):
            logger.error(f"Local image path does not exist: {local_image_path}")
            return JSONResponse({"message": "Local image path does not exist", "faces_detected": 0}, status_code=500)
        
@app.post("/live-portrait")
def live_portrait(request: LivePortraitRequestProto, api_key: str = Depends(header_scheme)):
    if live_portrait_pipeline is None:
        return JSONResponse({"message": "Server not ready", "output_path": ''}, status_code=503)
    if api_key != SERVING_API_KEY:
        return JSONResponse({"message": "Invalid API key", "output_path": ''}, status_code=401)
    
    src_input = None
    driving_input = None
    
    src_input, driving_input = extract_inputs_from_request(request)
    if src_input is None or driving_input is None:
        return JSONResponse({"message": "Failed to process inputs", "output_path": ''}, status_code=500)
    
    args = tyro.cli(ArgumentConfig)
    args.source = src_input
    args.driving = driving_input
    args.output_dir = OUTPUT_LOCAL_PATH
    wfp, wfp_concat = live_portrait_pipeline.execute(src_input, driving_input)
    print(wfp, wfp_concat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Portrait Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    args = parser.parse_args()

    # init_cos_client()
    # logger.info("COS client initialized")
    
    init_live_portrait_pipeline()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)