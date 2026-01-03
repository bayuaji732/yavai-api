import os
from dotenv import load_dotenv

load_dotenv()

YAVAI_API_BASE_URL = os.environ.get("YAVAI_API_BASE_URL")
    
# Database
PSQL_HOST = os.environ.get("PSQL_HOST")
PSQL_DATABASE = os.environ.get("PSQL_DATABASE")
PSQL_PORT = os.environ.get("PSQL_PORT")
PSQL_USER = os.environ.get("PSQL_USER")
PSQL_PASSWORD = os.environ.get("PSQL_PASSWORD")
    
# Redis
REDIS_HOST = os.environ.get("REDIS_HOST")
REDIS_PORT = os.environ.get("REDIS_PORT")
REDIS_DB = os.environ.get("REDIS_DB")
    
# Spark
SPARK_HOME = os.environ.get("SPARK_HOME")
SPARK_JARS = os.environ.get("SPARK_JARS")
SPARK_JARS_AWS = os.environ.get("SPARK_JARS_AWS")
SPARK_JARS_TFRECORD = os.environ.get("SPARK_JARS_TFRECORD")
SPARK_JARS_EXCEL = os.environ.get("SPARK_JARS_EXCEL")
SPARK_HADOOP_HIVE_METASTORE_URIS = os.environ.get("SPARK_HADOOP_HIVE_METASTORE_URIS")
SPARK_KERBEROS_KEYTAB = os.environ.get("SPARK_KERBEROS_KEYTAB")
SPARK_KERBEROS_PRINCIPAL = os.environ.get("SPARK_KERBEROS_PRINCIPAL")
SPARK_EXTRACLASSPATH = os.environ.get("SPARK_EXTRACLASSPATH")
    
# HDFS
HDFS_HOST = os.environ.get("HDFS_HOST")
HDFS_NAME_NODE = os.environ.get("HDFS_NAME_NODE")
HDFS_SERVER = os.environ.get("HDFS_SERVER")
HDFS_FUSE_MNT_DIR = os.environ.get("HDFS_FUSE_MNT_DIR")
HIVE_DIR_PATH = os.environ.get("HIVE_DIR_PATH")
HIVE_HIVESERVER2_JDBC_URL = os.environ.get("HIVE_HIVESERVER2_JDBC_URL")
HIVE_METASTORE_JARS = os.environ.get("HIVE_METASTORE_JARS")
    
# S3
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT")
    
# JWT
JWT_TOKEN_KEY = os.environ.get("JWT_TOKEN_KEY")

# DLIB Model
SHAPE_PREDICTOR_PATH= os.environ.get("SHAPE_PREDICTOR_PATH")
    
# File settings
URL: str = os.getenv("URL", "")
URL2: str = os.getenv("URL2", "")  # Feature group URL
URL3: str = os.getenv("URL3", "")  # Training dataset URL
LOCAL_DIR: str = os.getenv("LOCAL_DIR", "/tmp/dataprep")

# Hive settings (for feature group)
HIVE_HOST: str = os.getenv("HIVE_HOST", "localhost")
HIVE_PORT: int = int(os.getenv("HIVE_PORT", "10000"))
HIVE_PRINCIPAL: str = os.getenv("HIVE_PRINCIPAL", "")
HIVE_DATABASE: str = os.getenv("HIVE_DATABASE", "default")
    
# HDFS settings (for training datasets)
HADOOP_HOME: str = os.getenv("HADOOP_HOME", "/usr/hadoop")
HDFS_NAMENODE: str = os.getenv("HDFS_NAMENODE", "localhost")
TICKET_CACHE_PATH: str = os.getenv("TICKET_CACHE_PATH", "/tmp/krb5cc_1062")
    
# Logging settings
LOG_FILE_PATH: str = os.getenv("LOG_FILE_PATH", "logs/dataset.log")
MAX_LOG_SIZE: int = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT: int = 5
    
# API settings
MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100 MB
ALLOWED_FILE_TYPES: list = ['csv', 'tsv', 'xls', 'xlsx', 'sav']