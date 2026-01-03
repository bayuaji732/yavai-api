from contextlib import contextmanager
from pyhive import hive
from core import config

@contextmanager
def get_hive_connection():
    conn = hive.Connection(
        host=config.HIVE_HOST,
        port=int(config.HIVE_PORT),
        auth="KERBEROS",
        kerberos_service_name="hive",
        database=config.HIVE_DATABASE
    )
    try:
        yield conn
    finally:
        conn.close()