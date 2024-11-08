import argparse
import asyncio
import redis
import os
import time
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.live_portrait_pipeline import LivePortraitPipeline

OUTPUT_LOCAL_PATH = os.getenv("OUTPUT_LOCAL_PATH", "./tmp/outputs/")


# aioredis
# redis


import tyro

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


TASK_PREFIX = "lp-task-"  # Prefix for task keys, used for retrieving data/status
TASK_TAG = "lp"
STREAM_NAME_PREFIX = os.getenv("REDIS_STREAM_PREFIX", "task_stream") + "_"
CONSUMER_GROUP_PREFIX = os.getenv("REDIS_GROUP_PREFIX", "task_group") + "_"

LOCK_KEY_PREFIX = os.getenv("LOCK_KEY_PREFIX", "lock_key") + "_" + TASK_TAG
CONSUMER_NAME = os.getenv("HOSTNAME", "test_consumer_lp")

live_portrait_engine = None
r = None

def parse_args():
    parser = argparse.ArgumentParser(description="LivePortrait Redis Task Worker")
    parser.add_argument("--redis-host", type=str, default="43.156.39.249", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=31317, help="Redis port")
    return parser.parse_args()


def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})


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
    
    return live_portrait_pipeline


def task_worker(
    stream_name: str,
    consumer_group: str,
    consumer_name: str,
    task_prefix: str = "lp-task-",
    lock_key_prefix: str = "lock_key_lp",
    lock_timeout: int = 30,
):
    logger.info(f"Starting task worker {consumer_name} in {consumer_group} for stream {stream_name}...")
    
    while True:
        logger.info(f"Listening for tasks in {stream_name}...")
        
        tasks = r.xreadgroup(consumer_group, consumer_name, {stream_name: ">"}, count=1, block=0)
        
        if tasks:
            for stream_name, task_list in tasks:
                for task_id, task_data in task_list:

                    serving_task_id = task_data.get("request_id")
                    lock_key = lock_key_prefix + serving_task_id

                    try:
                        # 目前部署的redis - keydb是集群模式兼容单服务，存在多次消费，需要加锁
                        # Try to acquire the lock
                        lock_acquired = r.set(lock_key, consumer_name, nx=True, ex=lock_timeout)
                        if not lock_acquired:
                            logger.info(f"Failed to acquire lock {lock_key}. Skipping...")
                            continue
                        logger.info(
                            f"Acquired lock {lock_key} (expiring in {lock_timeout} secs), processing tasks from stream {stream_name}"
                            f"on {r.connection_pool.connection_kwargs['host']}:{r.connection_pool.connection_kwargs['port']}"
                        )

                        occupied_by_pod = r.hmget(task_prefix + serving_task_id, "occupied_by_pod")
                        if not occupied_by_pod:
                            logger.warning(
                                f"Task occupied_by_pod of {serving_task_id} not found. Processing the task anyway."
                            )
                        else:
                            occupied_by_pod = occupied_by_pod[0]
                            if occupied_by_pod and occupied_by_pod != consumer_name:
                                logger.error(
                                    f"Task {serving_task_id} is already being processed by another consumer: {occupied_by_pod}. Skipping..."
                                )
                                continue
                        # NOTE Important: mark the task as occupied by the current consumer
                        r.hmset(task_prefix + serving_task_id, {"occupied_by_pod": consumer_name})
                        logger.info(f"Set task {serving_task_id} as occupied by {consumer_name}")

                        task_payload = r.hgetall(task_prefix + serving_task_id)
                        logger.info(f"Processing task {serving_task_id}, task payload: {task_payload}")
                        if not task_payload:
                            raise RuntimeError(f"Task payload of {serving_task_id} not found")

                        # # decoded_data = {key.decode('utf-8'): value.decode('utf-8') for key, value in task_payload.items()}
                        # task_payload = task_payload.get("payload")  # actual payload from the request
                        # # res = await process_task(task_payload)
                        # # if not res:
                        # #     raise RuntimeError(f"Error processing task {serving_task_id}")
                        # logger.info(f"Processing task {serving_task_id} with payload {task_payload}...")
                        args = tyro.cli(ArgumentConfig)
                        args.source = task_payload.get("src_key")
                        args.driving = task_payload.get("driving_key")
                        args.output_dir = OUTPUT_LOCAL_PATH
                        wfp, wfp_concat = live_portrait_pipeline.execute(args)
                        print(wfp, wfp_concat)
                        live_portrait_engine

                    except Exception as e:
                        logger.error(f"{e}")
                        # Callback with failed status
                    finally:
                        # NOTE: messages are always acknowledged and deleted after the task finished or failed;
                        # We won't add back the task to the stream if it's failed
                        r.xack(stream_name, consumer_group, task_id)
                        r.xdel(stream_name, task_id)

                        if r.get(lock_key) == consumer_name:
                            r.delete(lock_key)
                            logger.info(f"Released lock {lock_key} by consumer {consumer_name}")

        time.sleep(1.0)

def run_worker(args):
    
    global live_portrait_engine, r
    
    redis_host, redis_port = args.redis_host, args.redis_port
    # task_group_lp
    consumer_group = CONSUMER_GROUP_PREFIX + TASK_TAG
    # task_stream_lp
    stream_name = STREAM_NAME_PREFIX + TASK_TAG
    
    r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
    logger.info(f"Creating consumer group {consumer_group} for stream {stream_name}")
    try:
        r.xgroup_create(stream_name, consumer_group, id="0", mkstream=True)
    except redis.exceptions.ResponseError as e:
        if "BUSYGROUP" in str(e):
            logger.warning(f"Consumer group {consumer_group} already exists")
        else:
            raise e
        
    logger.info("Initializing LivePortrait engine...")
    live_portrait_engine = init_live_portrait_pipeline()
    logger.info("LivePortrait engine initialized")
    
    task_worker(
        stream_name=stream_name,
        consumer_group=consumer_group,
        consumer_name=CONSUMER_NAME,
        task_prefix=TASK_PREFIX,
        lock_key_prefix=LOCK_KEY_PREFIX)
    
    
    
if __name__ == "__main__":
    args = parse_args()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_worker(args))