import psycopg2
from contextlib import contextmanager
from app.core import config

@contextmanager
def get_db_connection():
    conn = psycopg2.connect(
        host=config.PSQL_HOST,
        database=config.PSQL_DATABASE,
        port=config.PSQL_PORT,
        user=config.PSQL_USER,
        password=config.PSQL_PASSWORD
    )
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()