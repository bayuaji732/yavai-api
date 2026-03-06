import os
import findspark
from pyspark import SparkConf
from pyspark.sql import SparkSession
from app.core import config

if config.SPARK_HOME:
    findspark.init(config.SPARK_HOME)


def create_spark_session(app_name: str, hadoop_username: str = "apps") -> SparkSession:
    """
    Single Spark session supporting:
      - Existing Hive/Kerberos/HDFS cluster operations
      - Apache Iceberg via REST catalog at 172.20.3.111:8181
      - Data stored in Apache Ozone S3 (yavaidevwrk2.yava32.com:9878)

    Iceberg table reference pattern:
      {ICEBERG_CATALOG_NAME}.{ICEBERG_NAMESPACE}.{table_name}
      e.g.  rest.default.customer_features
    """
    os.environ["HADOOP_USER_NAME"] = hadoop_username

    catalog_name = config.ICEBERG_CATALOG_NAME   # "rest"
    catalog_uri  = config.ICEBERG_REST_URI        # "http://172.20.3.111:8181"
    warehouse    = config.ICEBERG_WAREHOUSE       # "s3a://iceberg/warehouse/"

    # Merge Iceberg runtime into the existing package list
    spark_packages = ",".join([
                config.SPARK_JARS_AWS,
                config.SPARK_JARS_TFRECORD,
                config.SPARK_JARS_EXCEL,
                config.SPARK_COMMONS_MATH,
                config.SPARK_JARS_ICEBERG,
])
    
    conf = (
        SparkConf()
        .setAppName(app_name)
        .set("spark.jars.packages", spark_packages)
        .set("spark.dynamicAllocation.enabled", "true")
        .set("spark.shuffle.service.enabled", "true")
        
        # ── Iceberg SQL extensions ──────────────────────────────────────────
        .set("spark.sql.extensions",
             "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")

        # ── Iceberg REST catalog (standalone at 172.20.3.111) ───────────────
        .set(f"spark.sql.catalog.{catalog_name}",
             "org.apache.iceberg.spark.SparkCatalog")
        .set(f"spark.sql.catalog.{catalog_name}.type",      "rest")
        .set(f"spark.sql.catalog.{catalog_name}.uri",        catalog_uri)
        .set(f"spark.sql.catalog.{catalog_name}.warehouse",  warehouse)
        # Ozone credentials for Iceberg data files
        .set(f"spark.sql.catalog.{catalog_name}.s3.access-key-id",     config.S3_ACCESS_KEY)
        .set(f"spark.sql.catalog.{catalog_name}.s3.secret-access-key", config.S3_SECRET_KEY)
        .set(f"spark.sql.catalog.{catalog_name}.s3.endpoint",          config.S3_ENDPOINT)
        .set(f"spark.sql.catalog.{catalog_name}.s3.path-style-access",  "true")

        # ── Hive / Metastore (existing, unchanged) ──────────────────────────
        .set("spark.hadoop.hive.metastore.uris", config.SPARK_HADOOP_HIVE_METASTORE_URIS)
        .set("spark.sql.hive.metastore.jars", config.HIVE_METASTORE_JARS)
        .set("spark.sql.hive.hiveserver2.jdbc.url", config.HIVE_HIVESERVER2_JDBC_URL)
        .set("spark.sql.dialect", "hiveql")
        .set("hive.strict.managed.tables", "false")
        .set("spark.hadoop.hive.strict.managed.tables", "false")
        .set("spark.hadoop.hive.execution.engine", "tez")
        .set("spark.hadoop.hive.vectorized.execution.enabled", "false")
        .set("spark.hadoop.hive.exec.stagingdir", "/tmp/hive-staging")
        .set("spark.datasource.hive.warehouse.load.staging.dir", "/tmp")

        # ── S3A / Ozone (shared – used by Hive external tables & Iceberg) ───
        .set("spark.hadoop.fs.s3a.impl", 
             "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .set("spark.hadoop.fs.s3a.access.key", config.S3_ACCESS_KEY)
        .set("spark.hadoop.fs.s3a.secret.key", config.S3_SECRET_KEY)
        .set("spark.hadoop.fs.s3a.endpoint", config.S3_ENDPOINT)
        .set("spark.hadoop.fs.s3a.aws.credentials.provider", 
             "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        .set("spark.hadoop.fs.s3a.path.style.access", "true")
        .set("spark.hadoop.fs.s3a.change.detection.mode", "none")
        .set("spark.hadoop.fs.s3a.change.detection.source", "etag")
        .set("spark.hadoop.fs.s3a.change.detection.version.required", "false")

        # ── HDFS (still used for source file reads / training datasets) ─────
        .set("spark.hadoop.fs.defaultFS", config.HDFS_NAME_NODE)
        .set("spark.hadoop.fs.hdfs.impl", 
             "org.apache.hadoop.hdfs.DistributedFileSystem")
        .set("spark.hadoop.fs.hdfs.server", config.HDFS_SERVER)
        
        # ── Security / Kerberos ─────────────────────────────────────────────
        .set("spark.kerberos.keytab", config.SPARK_KERBEROS_KEYTAB)
        .set("spark.kerberos.principal", config.SPARK_KERBEROS_PRINCIPAL)
        .set("spark.driver.extraJavaOptions", 
             "-Djavax.security.auth.useSubjectCredsOnly=false " 
             "-Djava.security.krb5.conf=/etc/krb5.conf")
        .set("spark.executor.extraJavaOptions", 
             "-Djavax.security.auth.useSubjectCredsOnly=false " 
             "-Djava.security.krb5.conf=/etc/krb5.conf")
        
        # ── Networking / RPC ────────────────────────────────────────────────
        .set("spark.dfs.data.transfer.protection", "privacy") 
        .set("spark.hadoop.rpc.protection", "privacy") 
        
        # ── Classpath ───────────────────────────────────────────────────────
        .set("spark.driver.extraClassPath", config.SPARK_EXTRACLASSPATH) 
        .set("spark.executor.extraClassPath", config.SPARK_EXTRACLASSPATH) 
        
        # ── Output ─────────────────────────────────────────────────────────
        .set("spark.hadoop.mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")
    )
        
        
    
    return (
        SparkSession.builder
        .master("local[*]")
        .appName(app_name)
        .config(conf=conf)
        .enableHiveSupport()
        .getOrCreate()
        )