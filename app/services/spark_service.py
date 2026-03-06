import re
import regex
import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import current_timestamp, date_format
from app.core import config


class SparkService:

    # ── Existing helpers (unchanged) ────────────────────────────────────────

    def pandas_to_spark(self, spark_session: SparkSession, df: pd.DataFrame) -> DataFrame:
        """Convert pandas DataFrame to Spark DataFrame"""
        return spark_session.createDataFrame(df)

    def read_csv(self, spark_session: SparkSession, path: str) -> DataFrame:
        """Read CSV file with delimiter detection"""
        header_list = spark_session.sparkContext.textFile(path).take(1)
        header_string = ''.join(header_list)
        result = re.search(r"([\t,;|])", header_string)
        delimiter = result.group() if result else ","
        return spark_session.read.options(
            header=True,
            delimiter=delimiter,
            escape='"',
            multiLine=True
        ).csv(path)

    def read_excel(self, spark_session: SparkSession, path: str) -> DataFrame:
        """Read Excel file"""
        return (spark_session.read
                .format("com.crealytics.spark.excel")
                .option("header", "true")
                .load(path))

    def add_timestamp_column(self, dataframe: DataFrame) -> DataFrame:
        """Add fg_date timestamp column"""
        df = dataframe.withColumn("fg_date", current_timestamp())
        df = df.withColumn("fg_date", date_format("fg_date", "yyyy-MM-dd'T'HH:mm:ss.SSSXXX"))
        return df

    # ── Hive (kept for backward compatibility / profiling reads) ────────────

    def save_to_hive(self, spark_session: SparkSession, dataframe: DataFrame, feature_group_object):
        """Save dataframe to Hive ORC table (legacy path, kept for profiling)."""
        hdfs_path = (
            f"{config.HDFS_NAME_NODE}/warehouse/tablespace/managed/hive"
            f"/{feature_group_object.table_name}"
        )
        dataframe = self.add_timestamp_column(dataframe)
        partition_keys = feature_group_object.partition_keys or []

        (dataframe.write
         .format("orc")
         .mode("overwrite")
         .partitionBy(partition_keys)
         .option("path", hdfs_path)
         .saveAsTable(feature_group_object.table_name))

    # ── Iceberg ─────────────────────────────────────────────────────────────

    def _iceberg_table_ref(self, table_name: str) -> str:
        """Build fully-qualified Iceberg table reference: catalog.namespace.table"""
        return f"{config.ICEBERG_CATALOG_NAME}.{config.ICEBERG_NAMESPACE}.{table_name}"

    def save_to_iceberg(
        self,
        spark_session: SparkSession,
        dataframe: DataFrame,
        feature_group_object,
    ) -> None:
        """
        Save a feature group DataFrame as an Iceberg table in the REST catalog.
        Data is stored in Ozone S3 (warehouse path from config.ICEBERG_WAREHOUSE).

        - Adds the fg_date timestamp column
        - Partitions by feature_group_object.partition_keys if provided
        - Uses createOrReplace so re-running the same feature group is idempotent
        """
        dataframe = self.add_timestamp_column(dataframe)
        full_table = self._iceberg_table_ref(feature_group_object.table_name)
        partition_keys = feature_group_object.partition_keys or []

        writer = dataframe.writeTo(full_table).using("iceberg")
        if partition_keys:
            writer = writer.partitionedBy(*partition_keys)
        writer.createOrReplace()

    def read_iceberg_table(
        self,
        spark_session: SparkSession,
        table_name: str,
        limit: int = None,
    ) -> DataFrame:
        """Read an Iceberg table (latest snapshot) as a Spark DataFrame."""
        full_table = self._iceberg_table_ref(table_name)
        query = f"SELECT * FROM {full_table}"
        if limit:
            query += f" LIMIT {limit}"
        return spark_session.sql(query)

    def drop_iceberg_table(
        self,
        spark_session: SparkSession,
        table_name: str,
        purge: bool = True,
    ) -> None:
        """
        Drop an Iceberg table from the REST catalog.
        purge=True (default) also removes the data files from Ozone S3.
        """
        full_table = self._iceberg_table_ref(table_name)
        purge_clause = "PURGE" if purge else ""
        spark_session.sql(f"DROP TABLE IF EXISTS {full_table} {purge_clause}")

    def save_training_dataset_iceberg(
        self,
        spark_session: SparkSession,
        training_dataset_object,
        dataframe: DataFrame,
    ) -> None:
        """
        Save a training dataset as an Iceberg table.
        The table name is derived from training_dataset_object.name (snake_case).
        training_dataset_object.path is set to the full Iceberg table ref so
        the existing path field still carries a meaningful locator.
        """
        table_name = (
            training_dataset_object.name
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )
        full_table = self._iceberg_table_ref(table_name)
        dataframe.writeTo(full_table).using("iceberg").createOrReplace()
        # Store the catalog ref in path so other services can locate the table
        training_dataset_object.path = full_table

    def read_training_dataset_iceberg(
        self,
        spark_session: SparkSession,
        training_dataset_object,
    ) -> DataFrame:
        """
        Read a training dataset from Iceberg.
        Falls back to HDFS/CSV path if the path is not an Iceberg ref.
        """
        path = getattr(training_dataset_object, "path", "")

        if path and path.startswith(config.ICEBERG_CATALOG_NAME + "."):
            # path was set by save_training_dataset_iceberg → is a table ref
            return spark_session.sql(f"SELECT * FROM {path}")

        # Legacy: path is an HDFS path (tfrecord or CSV)
        if training_dataset_object.dataset_format in {"tfrecord", "tfrecords"}:
            return (spark_session.read
                    .format("tfrecord")
                    .option("recordType", "Example")
                    .load(path))
        return self.read_csv(spark_session, path)

    def drop_training_dataset_iceberg(
        self,
        spark_session: SparkSession,
        training_dataset_object,
        purge: bool = True,
    ) -> None:
        """Drop a training dataset Iceberg table."""
        path = getattr(training_dataset_object, "path", "")

        if path and path.startswith(config.ICEBERG_CATALOG_NAME + "."):
            # Extract table name from full ref  rest.default.my_table → my_table
            table_name = path.split(".")[-1]
            self.drop_iceberg_table(spark_session, table_name, purge=purge)
        elif path and path.startswith("hdfs://"):
            # Legacy HDFS delete
            sc = spark_session.sparkContext
            fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(
                sc._jsc.hadoopConfiguration()
            )
            fs.delete(
                sc._jvm.org.apache.hadoop.fs.Path(path), True
            )

    # ── Legacy training dataset methods (kept for any callers that pass
    #    an explicit HDFS path, e.g. profiling service) ────────────────────

    def save_training_dataset(
        self,
        spark_session: SparkSession,
        training_dataset_object,
        dataframe: DataFrame,
    ):
        """Save training dataset to HDFS (legacy – use save_training_dataset_iceberg)."""
        if training_dataset_object.dataset_format in {"tfrecord", "tfrecords"}:
            (dataframe.write.mode("overwrite")
             .format("tfrecord")
             .option("recordType", "Example")
             .save(training_dataset_object.path))
        else:
            (dataframe.repartition(1).write.mode("overwrite")
             .format("csv")
             .option("header", True)
             .save(training_dataset_object.path))

    def read_training_dataset(
        self,
        spark_session: SparkSession,
        training_dataset_object,
    ) -> DataFrame:
        """Read training dataset from HDFS (legacy)."""
        if training_dataset_object.dataset_format in {"tfrecord", "tfrecords"}:
            return (spark_session.read
                    .format("tfrecord")
                    .option("recordType", "Example")
                    .load(training_dataset_object.path))
        return self.read_csv(spark_session, training_dataset_object.path)


# ── Utility functions (unchanged) ───────────────────────────────────────────

def camel_to_snake(name: str) -> str:
    result = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return result.lower()


class FeatureObject:
    def __init__(self, data):
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    setattr(self, key, FeatureObject(value))
                elif isinstance(value, list):
                    setattr(self, key, [
                        FeatureObject(item) if isinstance(item, dict) else item
                        for item in value
                    ])
                else:
                    setattr(self, key, value)
        else:
            self.__dict__ = data

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def to_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, FeatureObject):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [
                    item.to_dict() if isinstance(item, FeatureObject) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result


def convert_keys_to_snake_case(data):
    if isinstance(data, dict):
        return {camel_to_snake(k): convert_keys_to_snake_case(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_keys_to_snake_case(i) for i in data]
    return data


def parse_feature_group_json(feature_group_json: str):
    feature_group_json = replace_boolean_string(feature_group_json)
    feature_group_dict = eval(feature_group_json)
    feature_group_dict = convert_keys_to_snake_case(feature_group_dict)

    class FeatureGroupObject:
        def __init__(self, data):
            for key, value in data.items():
                if key == "features" and isinstance(value, list):
                    setattr(self, key, [
                        FeatureObject(item) if isinstance(item, dict) else item
                        for item in value
                    ])
                elif isinstance(value, dict):
                    setattr(self, key, FeatureObject(value))
                elif isinstance(value, list):
                    setattr(self, key, [
                        FeatureObject(item) if isinstance(item, dict) else item
                        for item in value
                    ])
                else:
                    setattr(self, key, value)

        def to_dict(self):
            result = {}
            for key, value in self.__dict__.items():
                if isinstance(value, FeatureObject):
                    result[key] = value.to_dict()
                elif isinstance(value, list):
                    result[key] = [
                        item.to_dict() if isinstance(item, FeatureObject) else item
                        for item in value
                    ]
                else:
                    result[key] = value
            return result

    return FeatureGroupObject(feature_group_dict)


def parse_training_dataset_json(training_dataset_json: str):
    training_dataset_dict = eval(training_dataset_json)

    class TrainingDatasetObject:
        def __init__(self, data):
            self.__dict__.update(data)

        def to_dict(self):
            return self.__dict__

    return TrainingDatasetObject(training_dataset_dict)


def replace_boolean_string(string: str) -> str:
    string = regex.sub(r"false", "False", string, flags=regex.UNICODE)
    string = regex.sub(r"true",  "True",  string, flags=regex.UNICODE)
    return string