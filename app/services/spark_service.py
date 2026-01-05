import re
import regex
import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import current_timestamp, date_format
from app.core import config
class SparkService:
    
    def pandas_to_spark(self, spark_session: SparkSession, df: pd.DataFrame) -> DataFrame:
        """Convert pandas DataFrame to Spark DataFrame"""
        return spark_session.createDataFrame(df)
    
    def read_csv(self, spark_session: SparkSession, path: str) -> DataFrame:
        """Read CSV file with delimiter detection"""
        header_list = spark_session.sparkContext.textFile(path).take(1)
        header_string = ''.join(header_list)
        
        result = re.search("([,;|])", header_string)
        delimiter = result.group() if result else ","
        
        return spark_session.read.options(
            header=True,
            delimiter=delimiter,
            escape='\"',
            multiLine=True
        ).csv(path)
    
    def read_excel(self, spark_session: SparkSession, path: str) -> DataFrame:
        """Read Excel file"""
        return spark_session.read.format("com.crealytics.spark.excel").option("header", "true").load(path)
    
    def add_timestamp_column(self, dataframe: DataFrame) -> DataFrame:
        """Add timestamp column"""
        df = dataframe.withColumn("fg_date", current_timestamp())
        df = df.withColumn("fg_date", date_format("fg_date", "yyyy-MM-dd'T'HH:mm:ss.SSSXXX"))
        return df
    
    def save_to_hive(self, spark_session: SparkSession, dataframe: DataFrame, feature_group_object):
        """Save dataframe to Hive table"""
        if feature_group_object.database_name:
            spark_session.sql(f"USE {feature_group_object.database_name}")
        
        hdfs_path = f"{config.HDFS_NAME_NODE}/warehouse/tablespace/managed/hive/{feature_group_object.table_name}"
        
        dataframe = self.add_timestamp_column(dataframe)
        
        partition_keys = feature_group_object.partition_keys if feature_group_object.partition_keys else []
        
        dataframe.write \
            .format("orc") \
            .mode("overwrite") \
            .partitionBy(partition_keys) \
            .option("path", hdfs_path) \
            .saveAsTable(feature_group_object.table_name)
    
    def save_training_dataset(self, spark_session: SparkSession, training_dataset_object, dataframe: DataFrame):
        """Save training dataset"""
        if training_dataset_object.dataset_format in {'tfrecord', 'tfrecords'}:
            dataframe.write.mode("overwrite").format("tfrecord").option("recordType", "Example").save(training_dataset_object.path)
        else:
            dataframe.repartition(1).write.mode("overwrite").format("csv").option("header", True).save(training_dataset_object.path)
    
    def read_training_dataset(self, spark_session: SparkSession, training_dataset_object) -> DataFrame:
        """Read training dataset"""
        if training_dataset_object.dataset_format in {'tfrecord', 'tfrecords'}:
            return spark_session.read.format("tfrecord").option("recordType", "Example").load(training_dataset_object.path)
        else:
            return self.read_csv(spark_session, training_dataset_object.path)

def parse_feature_group_json(feature_group_json: str):
    """Parse feature group JSON string"""
    feature_group_json = replace_boolean_string(feature_group_json)
    feature_group_dict = eval(feature_group_json)
    
    # Create simple object from dict
    class FeatureGroupObject:
        def __init__(self, data):
            self.__dict__.update(data)
        
        def to_dict(self):
            return self.__dict__
    
    return FeatureGroupObject(feature_group_dict)

def parse_training_dataset_json(training_dataset_json: str):
    """Parse training dataset JSON string"""
    training_dataset_dict = eval(training_dataset_json)
    
    class TrainingDatasetObject:
        def __init__(self, data):
            self.__dict__.update(data)
        
        def to_dict(self):
            return self.__dict__
    
    return TrainingDatasetObject(training_dataset_dict)

def replace_boolean_string(string: str) -> str:
    """Replace boolean strings with Python booleans"""
    string = regex.sub(r"false", "False", string, flags=regex.UNICODE)
    string = regex.sub(r"true", "True", string, flags=regex.UNICODE)
    return string