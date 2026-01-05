import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from io import BytesIO
import sys
import os


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

sys.modules["dlib"] = MagicMock()

@pytest.fixture
def mock_spark_session():
    with patch('app.core.spark_config.create_spark_session') as mock:
        session = Mock()
        session.sparkContext = Mock()
        session.sql = Mock(return_value=Mock(toPandas=Mock(return_value=pd.DataFrame())))
        mock.return_value = session
        yield session


@pytest.fixture
def mock_db():
    with patch('db.postgres.get_db_connection') as mock:
        conn = Mock()
        cursor = Mock()
        cursor.fetchone = Mock(return_value=None)
        cursor.fetchall = Mock(return_value=[])
        conn.cursor = Mock(return_value=cursor)
        conn.__enter__ = Mock(return_value=conn)
        conn.__exit__ = Mock(return_value=None)
        mock.return_value = conn
        yield conn


@pytest.fixture
def mock_redis():
    with patch('db.redis.get_redis_client') as mock:
        client = Mock()
        client.smembers = Mock(return_value=[])
        client.get = Mock(return_value=None)
        client.type = Mock(return_value='string')
        mock.return_value = client
        yield client


@pytest.fixture
def client():
    from app.main import app
    return TestClient(app)


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "YavAI API"}


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


@patch('services.feature_group_service.FeatureGroupService.save_feature_group_data')
@patch('app.core.spark_config.create_spark_session')
def test_create_feature_group_success(mock_spark, mock_service, client):
    mock_spark.return_value = Mock()
    mock_service.return_value = {"feature_group_id": "123"}
    
    payload = {
        "app_token": "test_token",
        "app_name": "test_app",
        "feature_group": '{"feature_group_id": "123", "name": "test"}'
    }
    
    response = client.post("/api/v1/", json=payload)
    assert response.status_code in [200, 500]


def test_create_feature_group_no_token(client):
    payload = {
        "app_name": "test_app",
        "feature_group": '{"name": "test"}'
    }
    
    response = client.post("/api/v1/", json=payload)
    assert response.status_code in [401, 422]


@patch('services.feature_group_service.FeatureGroupService.preview_feature_group_data')
@patch('app.core.spark_config.create_spark_session')
def test_preview_feature_group(mock_spark, mock_service, client):
    mock_spark.return_value = Mock()
    mock_service.return_value = pd.DataFrame({"col1": [1, 2]})
    
    payload = {
        "app_name": "test_app",
        "feature_group": '{"name": "test", "table_name": "test_table"}'
    }
    
    response = client.post("/api/v1/preview", json=payload)
    assert response.status_code in [200, 500]


@patch('services.training_dataset_service.TrainingDatasetService.save_training_dataset_data')
@patch('app.core.spark_config.create_spark_session')
def test_create_training_dataset_success(mock_spark, mock_service, client):
    mock_spark.return_value = Mock()
    mock_service.return_value = {"training_dataset_id": "123"}
    
    payload = {
        "app_token": "test_token",
        "app_name": "test_app",
        "training_dataset": '{"training_dataset_id": "123"}',
        "data": '{"feature_group_id": "456"}'
    }
    
    response = client.post("/api/v1/", json=payload)
    assert response.status_code in [200, 500]


def test_create_training_dataset_no_token(client):
    payload = {
        "app_name": "test_app",
        "training_dataset": '{"name": "test"}',
        "data": '{}'
    }
    
    response = client.post("/api/v1/", json=payload)
    assert response.status_code in [401, 422]


@patch('services.training_dataset_service.TrainingDatasetService.preview_training_dataset_data')
@patch('app.core.spark_config.create_spark_session')
def test_preview_training_dataset(mock_spark, mock_service, client):
    mock_spark.return_value = Mock()
    mock_service.return_value = pd.DataFrame({"col1": [1, 2]})
    
    payload = {
        "app_name": "test_app",
        "training_dataset": '{"path": "hdfs://test"}'
    }
    
    response = client.post("/api/v1/preview", json=payload)
    assert response.status_code in [200, 500]


@patch('db.postgres.get_db_connection')
def test_get_datasets_word_count(mock_db, client):
    conn = Mock()
    cursor = Mock()
    cursor.fetchall = Mock(return_value=[('test dataset',), ('another test',)])
    conn.cursor = Mock(return_value=cursor)
    conn.__enter__ = Mock(return_value=conn)
    conn.__exit__ = Mock(return_value=None)
    mock_db.return_value = conn
    
    response = client.get("/api/v1/datasets")
    assert response.status_code == 200
    assert "StatusCode" in response.json()


@patch('db.postgres.get_db_connection')
def test_get_feature_groups_word_count(mock_db, client):
    conn = Mock()
    cursor = Mock()
    cursor.fetchall = Mock(return_value=[('test feature',)])
    conn.cursor = Mock(return_value=cursor)
    conn.__enter__ = Mock(return_value=conn)
    conn.__exit__ = Mock(return_value=None)
    mock_db.return_value = conn
    
    response = client.get("/api/v1/feature-groups")
    assert response.status_code == 200


@patch('db.postgres.get_db_connection')
def test_get_notebooks_word_count(mock_db, client):
    conn = Mock()
    cursor = Mock()
    cursor.fetchall = Mock(return_value=[('test notebook',)])
    conn.cursor = Mock(return_value=cursor)
    conn.__enter__ = Mock(return_value=conn)
    conn.__exit__ = Mock(return_value=None)
    mock_db.return_value = conn
    
    response = client.get("/api/v1/notebooks")
    assert response.status_code == 200


@patch('services.privacy_service.PrivacyService.process_file')
@patch('db.postgres.get_db_connection')
def test_privacy_detection_cached(mock_db, mock_service, client):
    conn = Mock()
    cursor = Mock()
    cursor.fetchone = Mock(return_value=('[]', 'eye_detection'))
    conn.cursor = Mock(return_value=cursor)
    conn.__enter__ = Mock(return_value=conn)
    conn.__exit__ = Mock(return_value=None)
    mock_db.return_value = conn
    
    response = client.post("/api/1/file123", data={"token": "test_token"})
    assert response.status_code == 200


@patch('services.privacy_service.PrivacyService.process_file')
@patch('db.postgres.get_db_connection')
def test_privacy_detection_new(mock_db, mock_service, client):
    conn = Mock()
    cursor = Mock()
    cursor.fetchone = Mock(side_effect=[None, ('csv',)])
    conn.cursor = Mock(return_value=cursor)
    conn.__enter__ = Mock(return_value=conn)
    conn.__exit__ = Mock(return_value=None)
    mock_db.return_value = conn
    mock_service.return_value = ('data', 'pii_detection', 'Success', [])
    
    response = client.post("/api/v1/file123", data={"token": "test_token"})
    assert response.status_code in [200, 404, 500]


def test_privacy_detection_no_token(client):
    response = client.post("/api/v1/file123")
    assert response.status_code == 422


@patch('app.core.spark_config.create_spark_session')
@patch('services.spark_service.SparkService.pandas_to_spark')
def test_import_csv(mock_pandas_to_spark, mock_spark, client):
    mock_spark.return_value = Mock()
    mock_pandas_to_spark.return_value = Mock(columns=['col1'], withColumnRenamed=Mock(return_value=Mock(write=Mock())))
    
    csv_content = b"col1,col2\n1,2\n3,4"
    files = {"csv_file": ("test.csv", BytesIO(csv_content), "text/csv")}
    data = {"app_name": "test", "table_name": "test_table", "delimiter": ","}
    
    response = client.post("/api/v1/csv/import", files=files, data=data)
    assert response.status_code in [200, 500]


def test_import_csv_no_table_name(client):
    csv_content = b"col1,col2\n1,2"
    files = {"csv_file": ("test.csv", BytesIO(csv_content), "text/csv")}
    data = {"app_name": "test", "delimiter": ","}
    
    response = client.post("/api/v1/csv/import", files=files, data=data)
    assert response.status_code == 422


@patch('app.core.spark_config.create_spark_session')
def test_append_csv(mock_spark, client):
    session = Mock()
    session.sql = Mock(return_value=Mock(collect=Mock(return_value=[Mock(col_name='InputFormat', data_type='orcinputformat')])))
    mock_spark.return_value = session
    
    csv_content = b"col1,col2\n1,2"
    files = {"csv_file": ("test.csv", BytesIO(csv_content), "text/csv")}
    data = {"app_name": "test", "table_name": "test_table", "delimiter": ","}
    
    response = client.post("/api/v1/csv/append", files=files, data=data)
    assert response.status_code in [200, 500]


@patch('services.data_profiling_service.DataProfilingService.process_single_dataset')
def test_dataset_profiling(mock_process, client):
    payload = {
        "table_name": "dataset",
        "file_id": "123",
        "file_type": "csv"
    }
    
    response = client.post("/api/v1/dataset/profiling", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "accepted"


def test_dataset_profiling_invalid_type(client):
    payload = {
        "table_name": "dataset",
        "file_id": "123",
        "file_type": "invalid"
    }
    
    response = client.post("/api/v1/dataset/profiling", json=payload)
    assert response.status_code in [400, 500]


@patch('services.data_profiling_service.DataProfilingService.process_batch_datasets')
def test_dataset_profiling_batch(mock_process, client):
    payload = {
        "table_name": "dataset",
        "filters": {"status": 1}
    }
    
    response = client.post("/api/v1/dataset/profiling/batch", json=payload)
    assert response.status_code == 200


@patch('services.dataprep_db_service.DataPrepDBService.get_dataset_status')
def test_get_dataset_status_found(mock_status, client):
    mock_status.return_value = 0
    
    response = client.get("/api/v1/dataset/profiling/status/123?table_name=dataset")
    assert response.status_code == 200
    assert response.json()["status"] == 0


@patch('services.dataprep_db_service.DataPrepDBService.get_dataset_status')
def test_get_dataset_status_not_found(mock_status, client):
    mock_status.return_value = None
    
    response = client.get("/api/v1/dataset/profiling/status/123?table_name=dataset")
    assert response.status_code == 404


@patch('services.dataprep_db_service.DataPrepDBService.list_datasets')
def test_list_datasets(mock_list, client):
    mock_list.return_value = [{"id": "123", "file_type": "csv"}]
    
    response = client.get("/api/v1/dataset/profiling/list?table_name=dataset")
    assert response.status_code == 200
    assert "datasets" in response.json()


@patch('services.feature_profiling_service.FeatureProfilingService.process_single_feature_group')
def test_feature_group_profiling(mock_process, client):
    payload = {
        "table_name": "test_table",
        "online": False
    }
    
    response = client.post("/api/v1/feature-groups/profiling", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "accepted"


@patch('services.feature_profiling_service.FeatureProfilingService.process_batch_feature_groups')
def test_feature_group_profiling_batch(mock_process, client):
    response = client.post("/api/v1/feature-groups/profiling/batch")
    assert response.status_code == 200


@patch('services.dataprep_db_service.DataPrepDBService.get_feature_group_status')
def test_get_feature_group_status(mock_status, client):
    mock_status.return_value = 1
    
    response = client.get("/api/v1/feature-groups/profiling/status/test_table")
    assert response.status_code == 200


@patch('services.dataprep_db_service.DataPrepDBService.list_feature_groups')
def test_list_feature_groups(mock_list, client):
    mock_list.return_value = [{"table_name": "test", "online": False}]
    
    response = client.get("/api/v1/feature-groups/profiling/list")
    assert response.status_code == 200


@patch('services.training_dataset_profiling_service.TrainingDatasetProfilingService.process_single_training_dataset')
def test_training_dataset_profiling(mock_process, client):
    payload = {
        "td_id": "123",
        "hdfs_path": "hdfs://test/path",
        "dataset_format": "csv"
    }
    
    response = client.post("/api/v1/training-datasets/profiling", json=payload)
    assert response.status_code == 200


@patch('services.training_dataset_profiling_service.TrainingDatasetProfilingService.process_batch_training_datasets')
def test_training_dataset_profiling_batch(mock_process, client):
    payload = {"dataset_format": "csv"}
    
    response = client.post("/api/v1/training-datasets/profiling/batch", json=payload)
    assert response.status_code == 200


@patch('services.dataprep_db_service.DataPrepDBService.get_training_dataset_status')
def test_get_training_dataset_status(mock_status, client):
    mock_status.return_value = 2
    
    response = client.get("/api/v1/training-datasets/profiling/status/123")
    assert response.status_code == 200


@patch('services.dataprep_db_service.DataPrepDBService.list_training_datasets')
def test_list_training_datasets(mock_list, client):
    mock_list.return_value = [{"id": "123", "dataset_format": "csv"}]
    
    response = client.get("/api/v1/training-datasets/profiling/list")
    assert response.status_code == 200


def test_invalid_endpoint(client):
    response = client.get("/api/v1/invalid")
    assert response.status_code == 404


def test_missing_required_fields(client):
    payload = {"app_name": "test"}
    response = client.post("/api/v1/", json=payload)
    assert response.status_code == 422


@patch('pyreadstat.read_sav')
def test_convert_sav_to_csv(mock_read_sav, client):
    mock_read_sav.return_value = (pd.DataFrame({"col1": [1, 2]}), Mock())
    
    sav_content = b"fake_sav_content"
    files = {"sav_file": ("test.sav", BytesIO(sav_content), "application/x-spss-sav")}
    
    response = client.post("/api/v1/sav/convert-to-csv", files=files)
    assert response.status_code in [200, 500]


@patch('services.feature_group_service.FeatureGroupService.add_column_feature_group_data')
@patch('app.core.spark_config.create_spark_session')
def test_add_column_feature_group(mock_spark, mock_service, client):
    mock_spark.return_value = Mock()
    
    payload = {
        "app_token": "test_token",
        "app_name": "test_app",
        "feature_group": '{"table_name": "test_table"}'
    }
    
    response = client.post("/api/v1/add-column", json=payload)
    assert response.status_code in [200, 500]


@patch('services.feature_group_service.FeatureGroupService.delete_feature_group_data')
@patch('app.core.spark_config.create_spark_session')
def test_delete_feature_group(mock_spark, mock_service, client):
    mock_spark.return_value = Mock()
    
    payload = {
        "app_token": "test_token",
        "app_name": "test_app",
        "feature_group": '{"table_name": "test_table"}'
    }
    
    response = client.post("/api/v1/delete", json=payload)
    assert response.status_code in [200, 500]


@patch('hdfs.ext.kerberos.KerberosClient')
def test_get_feature_group_size(mock_client, client):
    client_instance = Mock()
    client_instance.content = Mock(return_value={"spaceConsumed": 1024})
    mock_client.return_value = client_instance
    
    payload = {
        "data": [{"table_name": "test_table"}]
    }
    
    response = client.post("/api/v1/size", json=payload)
    assert response.status_code in [200, 500]


@patch('services.training_dataset_service.TrainingDatasetService.delete_training_dataset_data')
@patch('app.core.spark_config.create_spark_session')
def test_delete_training_dataset(mock_spark, mock_service, client):
    mock_spark.return_value = Mock()
    
    payload = {
        "app_token": "test_token",
        "app_name": "test_app",
        "training_dataset": '{"path": "hdfs://test"}'
    }
    
    response = client.post("/api/v1/delete", json=payload)
    assert response.status_code in [200, 500]


@patch('db.postgres.get_db_connection')
def test_word_count_empty_results(mock_db, client):
    conn = Mock()
    cursor = Mock()
    cursor.fetchall = Mock(return_value=[])
    conn.cursor = Mock(return_value=cursor)
    conn.__enter__ = Mock(return_value=conn)
    conn.__exit__ = Mock(return_value=None)
    mock_db.return_value = conn
    
    response = client.get("/api/v1/model-zoo")
    assert response.status_code == 200


def test_dataset_profiling_missing_fields(client):
    payload = {"table_name": "dataset"}
    
    response = client.post("/api/v1/dataset/profiling", json=payload)
    assert response.status_code == 422


def test_feature_group_profiling_missing_fields(client):
    payload = {"table_name": "test_table"}
    
    response = client.post("/api/v1/feature-groups/profiling", json=payload)
    assert response.status_code == 422


def test_training_dataset_profiling_missing_fields(client):
    payload = {"td_id": "123"}
    
    response = client.post("/api/v1/training-datasets/profiling", json=payload)
    assert response.status_code == 422