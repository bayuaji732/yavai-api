from pyspark.sql import SparkSession
from typing import Any


class TrainingDatasetService:

    def save_training_dataset_data(
        self,
        spark_session: SparkSession,
        training_dataset_object: Any,
        training_dataset_dto_object: Any,
    ) -> dict:
        """
        Build query from feature group → read data → save as Iceberg table.
        training_dataset_object.path is set to the Iceberg table ref after save.
        """
        from app.services.spark_service import SparkService
        spark_service = SparkService()

        feature_group_obj = self._get_feature_group(
            training_dataset_dto_object.feature_group_id
        )
        query = self._build_query(feature_group_obj, training_dataset_dto_object)
        data = query.read(spark_session)

        # ── Save as Iceberg table (replaces HDFS CSV/tfrecord write) ───────
        spark_service.save_training_dataset_iceberg(
            spark_session, training_dataset_object, data
        )
        # training_dataset_object.path is now set to the Iceberg table ref

        return (
            training_dataset_object.to_dict()
            if hasattr(training_dataset_object, "to_dict")
            else {}
        )

    def preview_training_dataset_data(
        self,
        spark_session: SparkSession,
        training_dataset_object: Any,
    ):
        """
        Preview training dataset from Iceberg (or legacy HDFS path).
        Returns a pandas DataFrame (20 rows).
        """
        from app.services.spark_service import SparkService
        from app.core import config
        spark_service = SparkService()

        path = getattr(training_dataset_object, "path", "")

        if path and path.startswith(config.ICEBERG_CATALOG_NAME + "."):
            # Iceberg path — read directly from catalog
            df = spark_session.sql(f"SELECT * FROM {path} LIMIT 20")
            return df.toPandas()

        # Legacy HDFS path
        dataframe = spark_service.read_training_dataset(spark_session, training_dataset_object)
        return (
            dataframe.toPandas()
            if hasattr(dataframe, "toPandas")
            else dataframe
        )

    def delete_training_dataset_data(
        self,
        spark_session: SparkSession,
        training_dataset_object: Any,
    ):
        """
        Drop training dataset:
        - Iceberg table ref → drop table and purge data from Ozone S3
        - Legacy HDFS path → delete directory (backward compatibility)
        """
        from app.services.spark_service import SparkService
        from app.core import config
        spark_service = SparkService()

        path = getattr(training_dataset_object, "path", "")

        if not path:
            return

        if path.startswith(config.ICEBERG_CATALOG_NAME + "."):
            # Iceberg: drop table + purge Ozone S3 data files
            spark_service.drop_training_dataset_iceberg(
                spark_session, training_dataset_object, purge=True
            )
        elif path.startswith("hdfs://"):
            # Legacy HDFS delete
            sc = spark_session.sparkContext
            fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(
                sc._jsc.hadoopConfiguration()
            )
            fs.delete(
                sc._jvm.org.apache.hadoop.fs.Path(path), True
            )

    # ── Internal helpers ────────────────────────────────────────────────────

    def _get_feature_group(self, feature_group_id: str):
        """Fetch feature group metadata from dataset-management API."""
        import requests
        from app.core import config

        url = (
            f"{config.YAVAI_API_BASE_URL}/dataset-management/api/v1"
            f"/lib/feature-groups/{feature_group_id}"
        )
        response = requests.get(
            url,
            headers={"Content-Type": "application/json"},
            verify=False,
        )
        if response.status_code != 200:
            raise ValueError(f"Failed to get feature group: {response.status_code}")
        return response.json().get("data")

    def _build_query(self, feature_group_obj, training_dataset_dto_object):
        """
        Build a Spark SQL query against the feature group's Iceberg table.
        The feature group table ref: rest.default.<tableName>
        """
        from app.core import config

        table_name = feature_group_obj.get("tableName")
        iceberg_ref = (
            f"{config.ICEBERG_CATALOG_NAME}"
            f".{config.ICEBERG_NAMESPACE}"
            f".{table_name}"
        )

        class IcebergQuery:
            def read(self, spark_session):
                return spark_session.sql(f"SELECT * FROM {iceberg_ref}")

        return IcebergQuery()

    def _create_path(self, training_dataset_object) -> str:
        """
        Legacy: returns an HDFS path. Not used when saving to Iceberg.
        Kept in case any external caller still references this method.
        """
        import os
        from app.core import config

        path = os.path.join(
            config.HDFS_NAME_NODE, "user", "apps", "hive"
        )
        return os.path.join(
            path,
            f"training_dataset_{training_dataset_object.training_dataset_id}"
        )