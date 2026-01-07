import os
import findspark
from pyspark import SparkConf
from pyspark.sql import SparkSession
from app.core import config

if config.SPARK_HOME:
    findspark.init(config.SPARK_HOME)

def create_spark_session(app_name: str, hadoop_username: str = "apps") -> SparkSession:
    os.environ["HADOOP_USER_NAME"] = hadoop_username

    spark_packages = ",".join(
            [
                config.SPARK_JARS_AWS,
                config.SPARK_JARS_TFRECORD,
                config.SPARK_JARS_EXCEL,
            ]
        )
    
    conf = (
        SparkConf()
        .setAppName(app_name)
        .set("spark.jars", config.SPARK_JARS)
        .set("spark.jars.packages", spark_packages)
        .set("spark.dynamicAllocation.enabled", "true")
        .set("spark.shuffle.service.enabled", "true")
        # Hive / Metastore
        .set("spark.hadoop.hive.metastore.uris", config.SPARK_HADOOP_HIVE_METASTORE_URIS)
        .set("spark.sql.hive.metastore.jars", config.HIVE_METASTORE_JARS)
        .set("spark.sql.hive.hiveserver2.jdbc.url", config.HIVE_HIVESERVER2_JDBC_URL)
        .set("spark.sql.dialect", "hiveql")
        .set("hive.strict.managed.tables", "false")
        .set("spark.hadoop.hive.strict.managed.tables", "false")
        # Hive execution
        .set("spark.hadoop.hive.execution.engine", "tez")
        .set("spark.hadoop.hive.vectorized.execution.enabled", "false")
        .set("spark.hadoop.hive.exec.stagingdir", "/tmp/hive-staging")
        .set("spark.datasource.hive.warehouse.load.staging.dir", "/tmp")
        # S3A
        .set("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .set("spark.hadoop.fs.s3a.access.key", config.S3_ACCESS_KEY)
        .set("spark.hadoop.fs.s3a.secret.key", config.S3_SECRET_KEY)
        .set("spark.hadoop.fs.s3a.endpoint", config.S3_ENDPOINT)
        .set("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        .set("spark.hadoop.fs.s3a.path.style.access", "true")
        .set("spark.hadoop.fs.s3a.change.detection.mode", "none")
        .set("spark.hadoop.fs.s3a.change.detection.source", "etag")
        .set("spark.hadoop.fs.s3a.change.detection.version.required", "false")
        # HDFS
        .set("spark.hadoop.fs.defaultFS", config.HDFS_NAME_NODE)
        .set("spark.hadoop.fs.hdfs.impl", "org.apache.hadoop.hdfs.DistributedFileSystem")
        .set("spark.hadoop.fs.hdfs.server", config.HDFS_SERVER)
        # Security / Kerberos
        .set("spark.kerberos.keytab", config.SPARK_KERBEROS_KEYTAB)
        .set("spark.kerberos.principal", config.SPARK_KERBEROS_PRINCIPAL)
        .set("spark.driver.extraJavaOptions", "-Djavax.security.auth.useSubjectCredsOnly=false -Djava.security.krb5.conf=/etc/krb5.conf")
        .set("spark.executor.extraJavaOptions", "-Djavax.security.auth.useSubjectCredsOnly=false -Djava.security.krb5.conf=/etc/krb5.conf")
        # Networking / RPC
        .set("spark.dfs.data.transfer.protection", "privacy") 
        .set("spark.hadoop.rpc.protection", "privacy") 
        # Classpath
        .set("spark.driver.extraClassPath", config.SPARK_EXTRACLASSPATH) 
        .set("spark.executor.extraClassPath", config.SPARK_EXTRACLASSPATH) 
        # Output
        .set("spark.hadoop.mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")
    )
        
        
    
    return SparkSession.builder \
        .master("local[*]") \
        .appName(app_name) \
        .config(conf=conf) \
        .enableHiveSupport() \
        .getOrCreate()