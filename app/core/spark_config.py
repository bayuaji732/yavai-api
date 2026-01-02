import os
import findspark
from pyspark import SparkConf
from pyspark.sql import SparkSession
from core import config

findspark.init(config.SPARK_HOME)

def create_spark_session(app_name: str, hadoop_username: str = "apps") -> SparkSession:
    os.environ["HADOOP_USER_NAME"] = hadoop_username
    
    conf = SparkConf() \
        .set("spark.jars", config.SPARK_JARS) \
        .set("spark.dynamicAllocation.enabled", "true") \
        .set("spark.shuffle.service.enabled", "true") \
        .set("spark.hadoop.hive.metastore.uris", config.SPARK_HADOOP_HIVE_METASTORE_URIS) \
        .set("spark.datasource.hive.warehouse.metastoreUri", config.SPARK_HADOOP_HIVE_METASTORE_URIS) \
        .set("spark.hadoop.hive.execution.engine", "tez") \
        .set("spark.hadoop.hive.vectorized.execution.enabled", "false") \
        .set("spark.datasource.hive.warehouse.load.staging.dir", "/tmp") \
        .set("spark.hadoop.hive.exec.stagingdir", "/tmp/hive-staging") \
        .set("spark.hadoop.fs.s3a.access.key", config.S3_ACCESS_KEY) \
        .set("spark.hadoop.fs.s3a.secret.key", config.S3_SECRET_KEY) \
        .set("spark.hadoop.fs.s3a.endpoint", config.S3_ENDPOINT) \
        .set("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .set("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider") \
        .set("spark.hadoop.fs.s3a.path.style.access", "true") \
        .set("spark.hadoop.fs.s3a.change.detection.version.required", "false") \
        .set("spark.jars.packages", config.SPARK_JARS_AWS) \
        .set("spark.kerberos.keytab", config.SPARK_KERBEROS_KEYTAB) \
        .set("spark.kerberos.principal", config.SPARK_KERBEROS_PRINCIPAL) \
        .set("spark.driver.extraJavaOptions", "-Djavax.security.auth.useSubjectCredsOnly=false -Djava.security.krb5.conf=/etc/krb5.conf") \
        .set("spark.executor.extraJavaOptions", "-Djavax.security.auth.useSubjectCredsOnly=false -Djava.security.krb5.conf=/etc/krb5.conf") \
        .set("spark.sql.hive.hiveserver2.jdbc.url", config.HIVE_HIVESERVER2_JDBC_URL) \
        .set("spark.sql.hive.metastore.jars", config.HIVE_METASTORE_JARS) \
        .set("spark.driver.extraClassPath", config.SPARK_EXTRACLASSPATH) \
        .set("spark.executor.extraClassPath", config.SPARK_EXTRACLASSPATH) \
        .set("spark.hadoop.fs.defaultFS", config.HDFS_NAME_NODE) \
        .set("spark.hadoop.fs.hdfs.impl", "org.apache.hadoop.hdfs.DistributedFileSystem") \
        .set("spark.hadoop.fs.hdfs.server", config.HDFS_SERVER) \
        .set("spark.sql.dialect", "hiveql") \
        .set("hive.strict.managed.tables", "false") \
        .set("spark.hadoop.hive.strict.managed.tables", "false")
    
    return SparkSession.builder \
        .master("local[*]") \
        .appName(app_name) \
        .config(conf=conf) \
        .enableHiveSupport() \
        .getOrCreate()