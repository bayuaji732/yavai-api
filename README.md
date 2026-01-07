# YavAI API - FastAPI

Modern FastAPI implementation of the YavAI API with comprehensive data processing and profiling capabilities.

## Project Structure

```
fastapi_app/
├── main.py                 # Application entry point
├── api/                    # API routes and dependencies
│   ├── routes/            # Endpoint definitions
│   └── dependencies.py    # Dependency injection
├── core/                   # Configuration and security
│   ├── config.py          # Environment configuration
│   ├── security.py        # JWT authentication
│   ├── spark_config.py    # Spark session management
│   └── utils.py           # Utility functions
├── models/                 # Pydantic models
├── services/               # Business logic
│   ├── feature_group_service.py
│   ├── training_dataset_service.py
│   ├── analytics_service.py
│   ├── privacy_service.py
│   ├── spark_service.py
│   ├── dataprep_db_service.py
│   ├── data_profiling_service.py
│   ├── feature_profiling_service.py
│   └── training_dataset_profiling_service.py
├── db/                     # Database connections
│   ├── postgres.py
│   ├── redis.py
│   └── hive.py
└── utils/                  # Utility functions
```

## Features

- **Feature Groups**: Manage and process feature groups with Spark
- **Training Datasets**: Create and manage training datasets
- **Privacy Detection**: Detect PII in data and faces/eyes in images
- **Data Profiling**: Generate comprehensive data quality reports using ydata-profiling
- **Data Analytics**: Word frequency analysis across datasets
- **CSV Import**: Import and transform CSV files to Hive tables

## Installation

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Apache Spark (configured via environment)
- Hadoop/HDFS (for big data operations)
- PostgreSQL
- Redis

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/bayuaji732/yavai-api.git
cd yavai_api
```

2. **Install Python dependencies**

```bash
pip install -r requirements.txt
```

3. **Download required models**

```bash
# Download spaCy model for NLP/PII detection
python -m spacy download en_core_web_sm

# Download dlib face landmark predictor (optional, for face detection)
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
mv shape_predictor_68_face_landmarks.dat /app/ml_models/
```

4. **Configure environment**

```bash
cp .env.example .env
# Edit .env with your configuration
```

## Running the Application

### Using Docker (Recommended)

```bash
docker-compose up --build
```

The API will be available at `http://localhost:3304`

### Local Development

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 3304
```

### Production

```bash
uvicorn main:app --host 0.0.0.0 --port 3304 --workers 4
```

## Logging

The application uses standard Python logging that outputs to stdout/stderr, making it ideal for Docker and container orchestration platforms.

### Logging Configuration

- **Development**: Logs are output to console with INFO level by default
- **Docker**: Logs are captured by Docker and can be viewed with `docker logs <container-id>`
- **Production**: Configure log level via Uvicorn flags:
  ```bash
  uvicorn main:app --log-level info
  ```

### Viewing Logs

```bash
# Docker Compose
docker-compose logs -f api

# Docker
docker logs -f <container-id>

# Local development
# Logs appear directly in your terminal
```

### Log Levels

- `DEBUG`: Detailed information for debugging
- `INFO`: General operational messages (default)
- `WARNING`: Warning messages for potential issues
- `ERROR`: Error messages for failures

To change log level in Docker, update the `docker-compose.yml`:

```yaml
command: uvicorn main:app --host 0.0.0.0 --port 3304 --log-level debug
```

## API Documentation

Once running, access the interactive API documentation:

- **Swagger UI**: http://localhost:3304/docs
- **ReDoc**: http://localhost:3304/redoc

## Endpoints Overview

### Feature Groups

- `POST /api/v1/feature-groups` - Create feature group data
- `POST /api/v1/feature-groups/preview` - Preview feature group
- `POST /api/v1/feature-groups/download` - Download as CSV
- `POST /api/v1/feature-groups/delete` - Delete feature group
- `POST /api/v1/feature-groups/size` - Get storage size

### Training Datasets

- `POST /api/v1/training-datasets` - Create training dataset
- `POST /api/v1/training-datasets/preview` - Preview dataset
- `POST /api/v1/training-datasets/delete` - Delete dataset

### Data Profiling

- `POST /api/v1/dataprep/dataset/profiling` - Process single dataset
- `POST /api/v1/dataprep/dataset/profiling/batch` - Batch process datasets
- `GET /api/v1/dataprep/dataset/profiling/status/{file_id}` - Get processing status
- `GET /api/v1/dataprep/dataset/profiling/list` - List datasets

### Feature Store Profiling

- `POST /api/v1/dataprep/feature-groups/profiling` - Process feature group
- `POST /api/v1/dataprep/feature-groups/profiling/batch` - Batch process feature groups
- `GET /api/v1/dataprep/feature-groups/profiling/status/{table_name}` - Get status
- `GET /api/v1/dataprep/feature-groups/profiling/list` - List feature groups

### Training Dataset Profiling

- `POST /api/v1/dataprep/training-dataset/profiling` - Process training dataset
- `POST /api/v1/dataprep/training-dataset/profiling/batch` - Batch process training datasets
- `GET /api/v1/dataprep/training-dataset/profiling/status/{td_id}` - Get status
- `GET /api/v1/dataprep/training-dataset/profiling/list` - List training datasets

### Analytics

- `GET /api/v1/word-count/datasets` - Dataset name word frequency
- `GET /api/v1/word-count/feature-groups` - Feature store word frequency
- `GET /api/v1/word-count/notebooks` - Notebook name word frequency
- `GET /api/v1/word-count/file-management` - File management word frequency
- `GET /api/v1/word-count/model-zoo` - Model zoo word frequency

### Privacy Detection

- `POST /api/v1/privacy-detection/` - Detect PII/faces in files

### CSV Import

- `POST /api/v1/ingestion/csv/import` - Import CSV to new Hive table
- `POST /api/v1/ingestion/csv/append` - Insert CSV to existing table
- `POST /api/v1/ingestion/sav/convert-to-csv` - Convert SPSS .sav to CSV

## Environment Variables

See `.env.example` for all required configuration options. Key variables include:

### Core Configuration

- `YAVAI_API_BASE_URL` - Base URL for YavAI API
- `LOCAL_DIR` - Local directory for temporary files (default: `/tmp/dataprep`)

### Database

- `PSQL_HOST`, `PSQL_DATABASE`, `PSQL_PORT`, `PSQL_USER`, `PSQL_PASSWORD` - PostgreSQL connection

### Redis

- `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB` - Redis connection

### Spark & Hadoop

- `SPARK_HOME` - Spark installation directory
- `HDFS_NAME_NODE` - HDFS namenode URL
- `HIVE_HOST`, `HIVE_PORT` - Hive connection

### Authentication

- `JWT_TOKEN_KEY` - Base64-encoded secret key for JWT

## Development

### Running Tests

```bash
pytest
```

### Code Quality

```bash
# Linting
flake8 app/

# Type checking
mypy app/

# Formatting
black app/
```

### Adding New Endpoints

1. Create route file in `app/api/routes/`
2. Add business logic in `app/services/`
3. Define Pydantic models in `app/models/`
4. Register router in `app/main.py`

## Docker Configuration

The application is containerized with the following considerations:

- Logs output to stdout/stderr for Docker log drivers
- Environment variables loaded from `.env` or Docker secrets
- Health check endpoint at `/health`
- Graceful shutdown handling

### Docker Compose Services

```yaml
services:
  api:
    build: .
    ports:
      - "3304:3304"
    env_file:
      - .env
    volumes:
      - ./:/app
```

## Troubleshooting

### Common Issues

1. **Spark Connection Errors**

   - Verify Spark is running: `spark-shell`
   - Check `SPARK_HOME` environment variable
   - Ensure Hadoop/HDFS is accessible

2. **Database Connection Errors**

   - Verify PostgreSQL is running
   - Check connection credentials in `.env`
   - Ensure database exists

3. **Import Errors**

   - Reinstall dependencies: `pip install -r requirements.txt`
   - Download missing models (spaCy, dlib)

4. **Privacy Detection Not Working**
   - Verify spaCy model is installed: `python -m spacy download en_core_web_sm`
   - Check dlib model path in config

### Debugging

Enable debug logging by setting:

```bash
# In docker-compose.yml
command: uvicorn main:app --host 0.0.0.0 --port 3304 --log-level debug

# Or locally
uvicorn main:app --reload --log-level debug
```

## Performance Optimization

- Use connection pooling for database connections
- Enable Spark dynamic allocation for better resource usage
- Configure Redis for caching frequently accessed data
- Use background tasks for long-running operations

## Security

- JWT-based authentication for all sensitive endpoints
- Input validation using Pydantic models
- SQL injection prevention via parameterized queries
- Environment-based secrets management

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT

## Support

For issues and questions:

- Create an issue on GitHub
- Check existing documentation
- Review API documentation at `/docs`
