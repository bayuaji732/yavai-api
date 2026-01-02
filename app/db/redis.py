import redis
from core import config

def get_redis_client():
    return redis.StrictRedis(
        host=config.REDIS_HOST,
        port=config.REDIS_PORT,
        db=config.REDIS_DB,
        decode_responses=True
    )