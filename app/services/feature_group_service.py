import os
import regex
from typing import Any
from pyspark.sql import SparkSession
from app.core import config


class FeatureGroupService:

    def save_feature_group_data(
        self, spark_session: SparkSession, feature_group_object: Any
    ) -> dict:
        """
        Read source file → process → save as Iceberg table in REST catalog.
        Data is written to Ozone S3 (ICEBERG_WAREHOUSE).
        Table is registered under: rest.default.<table_name>
        """
        from app.services.spark_service import SparkService
        spark_service = SparkService()

        # Get source file path (S3A path from dataset-management API)
        file_path = self._get_file_path(feature_group_object.file_item_id)
        if not file_path:
            raise ValueError("Cannot write feature group data: file path is empty")

        # Read source file
        if feature_group_object.data_type in ["xls", "xlsx"]:
            spark_dataframe = spark_service.read_excel(spark_session, file_path)
        elif feature_group_object.data_type in ["csv", "tsv"]:
            spark_dataframe = spark_service.read_csv(spark_session, file_path)
        else:
            raise ValueError("Data types not supported. Use xls/xlsx or csv/tsv format")

        # Clean column names
        for column_name in spark_dataframe.columns:
            new_column_name = regex.sub(r"\s+", "_", column_name)
            new_column_name = regex.sub(r"[^A-Za-z0-9_]", "", new_column_name)
            new_column_name = new_column_name.lower()
            spark_dataframe = spark_dataframe.withColumnRenamed(column_name, new_column_name)

        # Select and extract features
        spark_dataframe = self._select_columns(spark_dataframe, feature_group_object)
        spark_dataframe = self._extract_features(spark_session, spark_dataframe, feature_group_object)

        # ── Save as Iceberg table (replaces save_to_hive) ──────────────────
        spark_service.save_to_iceberg(spark_session, spark_dataframe, feature_group_object)

        return (
            feature_group_object.to_dict()
            if hasattr(feature_group_object, "to_dict")
            else {}
        )

    def preview_feature_group_data(
        self, spark_session: SparkSession, feature_group_object: Any
    ):
        """
        Preview data from the Iceberg table.
        Online (Redis) feature groups are unchanged.
        """
        if not feature_group_object.table_name or feature_group_object.status != "SUCCESS":
            raise ValueError("Feature group currently has no data")

        from app.services.spark_service import SparkService
        spark_service = SparkService()

        columns = self._get_columns_to_extract(feature_group_object)
        extracted_columns = [self._clean_column_name(col) for col in columns]

        if feature_group_object.feature_group_online:
            # Online path stays as Hive / Redis (unchanged)
            full_table = feature_group_object.table_name
        else:
            # Offline path → Iceberg
            full_table = spark_service._iceberg_table_ref(feature_group_object.table_name)

        # Check if fg_date exists
        df_check = spark_session.sql(f"SELECT * FROM {full_table} LIMIT 1")
        if "fg_date" in df_check.columns:
            columns_result = ", ".join(extracted_columns + ["fg_date"])
        else:
            columns_result = ", ".join(extracted_columns)

        return (
            spark_session.sql(
                f"SELECT {columns_result} FROM {full_table} LIMIT 20"
            ).toPandas()
        )

    def download_feature_group_data(
        self,
        spark_session: SparkSession,
        feature_group_object: Any,
        destination_path: str,
    ) -> str:
        """Download full feature group from Iceberg as CSV."""
        from app.services.spark_service import SparkService
        spark_service = SparkService()

        if feature_group_object.feature_group_online:
            full_table = feature_group_object.table_name
        else:
            full_table = spark_service._iceberg_table_ref(feature_group_object.table_name)

        pandas_df = (
            spark_session.sql(f"SELECT * FROM {full_table}")
            .toPandas()
        )

        os.makedirs(destination_path, exist_ok=True)
        local_file_path = os.path.join(
            destination_path, f"{feature_group_object.feature_group_id}.csv"
        )
        pandas_df.to_csv(local_file_path, index=False)
        return local_file_path

    def add_column_feature_group_data(
        self, spark_session: SparkSession, feature_group_object: Any
    ):
        """Add columns to the Iceberg table using ALTER TABLE."""
        from app.services.spark_service import SparkService
        spark_service = SparkService()

        full_table = spark_service._iceberg_table_ref(feature_group_object.table_name)
        columns = self._get_columns_to_extract(feature_group_object)
        col_defs = ", ".join(
            [f"{self._clean_column_name(col)} string" for col in columns]
        )
        spark_session.sql(f"ALTER TABLE {full_table} ADD COLUMNS ({col_defs})")

    def delete_feature_group_data(
        self, spark_session: SparkSession, feature_group_object: Any
    ):
        """
        Drop the Iceberg table and purge its data files from Ozone S3.
        Online feature groups (Redis) still use the existing Hive DROP.
        """
        from app.services.spark_service import SparkService
        spark_service = SparkService()

        if feature_group_object.feature_group_online:
            # Online path: drop from Hive (unchanged)
            spark_session.sql(
                f"DROP TABLE IF EXISTS {feature_group_object.table_name}"
            )
        else:
            # Offline path: drop from Iceberg + purge data in Ozone
            spark_service.drop_iceberg_table(
                spark_session,
                feature_group_object.table_name,
                purge=True,
            )

    # ── Internal helpers (unchanged) ────────────────────────────────────────

    def _get_file_path(self, file_item_id: str) -> str:
        import requests
        url = (
            f"{config.YAVAI_API_BASE_URL}/dataset-management/api/v1"
            f"/lib/files/{file_item_id}/s3a-path"
        )
        response = requests.get(
            url, headers={"Content-Type": "application/json"}, verify=False
        )
        if response.status_code != 200:
            raise ValueError(f"Failed to get file path: {response.status_code}")
        return response.json().get("data")

    def _select_columns(self, dataframe, feature_group_object):
        columns = self._get_columns_to_extract(feature_group_object)
        return dataframe.select(*columns)

    def _extract_features(self, spark_session, dataframe, feature_group_object):
        # Placeholder for TF-IDF, OneHot, TextCleansing etc.
        return dataframe

    def _get_columns_to_extract(self, feature_group_object) -> list:
        return [feature.name for feature in feature_group_object.features]

    def _clean_column_name(self, column_name: str) -> str:
        new_name = regex.sub(r"\s+", "_", column_name)
        new_name = regex.sub(r"[^A-Za-z0-9_]", "", new_name)
        return new_name.lower()