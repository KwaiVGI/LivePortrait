import os
import argparse
import logging
from fastapi.security import APIKeyHeader
from fastapi import FastAPI, Depends
from typing import Optional, Tuple
import aioredis

import uvicorn

from pydantic import BaseModel, Field, root_validator
from starlette.responses import JSONResponse

TASK_PREFIX = "lp-task-"  # Prefix for task keys, used for retrieving data/status
TASK_TAG = "lp"
STREAM_NAME_PREFIX = os.getenv("REDIS_STREAM_PREFIX", "task_stream") + "_"
CONSUMER_GROUP_PREFIX = os.getenv("REDIS_GROUP_PREFIX", "task_group") + "_"
TASK_STATUS_EXPIRE = int(os.getenv("TASK_STATUS_EXPIRE", 43200))  # 12 hours by default


import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


r = None

async def _init_consumer_group(stream_name: str, group_name: str):
    try:
        await r.xgroup_create(stream_name, group_name, id="0", mkstream=True)
    except aioredis.exceptions.ResponseError as e:
        if "BUSYGROUP" in str(e):
            logger.warning(f"Consumer group {group_name} already exists in {stream_name}")
        else:
            raise e
        
        
async def lifespan(app: FastAPI):
    global r
    r = aioredis.Redis(host=args.redis_host, port=args.redis_port, decode_responses=True)
    
    await _init_consumer_group(STREAM_NAME_PREFIX + TASK_TAG, CONSUMER_GROUP_PREFIX + TASK_TAG)
    
    yield
    await r.close()
    logger.info("Redis connection closed")
    logger.info("Proxy server shut down")
    
app = FastAPI(lifespan=lifespan)


class LivePortraitRequestProto(BaseModel):
    '''Define the request schema for the Live Portrait API.
    Live Portrait needs two inputs to generate the final output:
    1. The source image/video (src_key or src_local_path)
    2. The driving image/video (driving_key or driving_local_path)
    This server will process source input according to the driving input.
    Some characteristics of the driving input will be transferred and applied to the source input.
    '''
    request_id: str = Field('test_request_id_0', description="Unique request ID", min_length=1)
    src_key: Optional[str] = Field(None, description="Object key of the source image/video from COS", min_length=1)
    driving_key: Optional[str] = Field(None, description="Object key of the driving image/video from COS", min_length=1)

    @root_validator(pre=True)
    def check_either_image_key_or_local_path(cls, values):
        src_key = values.get("src_key")
        driving_key = values.get("driving_key")
        if not src_key or not driving_key:
            raise ValueError("The 'src_key' and 'driving_key' must be provided")
        return values
    

@app.post("/submit")
async def submit(req: LivePortraitRequestProto):
    mapping = {
        "src_key": req.src_key,
        "driving_key": req.driving_key,
    }

    # Task data/status must be set first before adding to the queue
    task_name = TASK_PREFIX + req.request_id
    
    logger.info(f"Recving task {task_name} with payload {mapping}")
    
    n = await r.hset(task_name, mapping=mapping)
    await r.expire(task_name, TASK_STATUS_EXPIRE)
    logger.info(f"{n} fields successfully added for {task_name} (expire in {TASK_STATUS_EXPIRE} secs)")

    # Example: task_stream_lp
    queue_name = STREAM_NAME_PREFIX + TASK_TAG
    redis_task_id = await r.xadd(queue_name, {"request_id": req.request_id})
    logger.info(f"Redis task {redis_task_id} added to Redis Stream {queue_name}")

    # trim the Redis stream
    await r.xtrim(queue_name, maxlen=2000, approximate=True)

    return JSONResponse(content={"request_id": req.request_id, "message_id": redis_task_id})


def parse_args():
    parser = argparse.ArgumentParser(description="Proxy server for OpenSora serving")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--redis-host", type=str, default="43.156.39.249", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=31317, help="Redis port")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
    logger.info(f"Proxy server is running on {args.host}:{args.port})")