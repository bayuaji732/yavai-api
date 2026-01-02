# YavAI Feature Store API - FastAPI

Modern FastAPI implementation of the YavAI Feature Store API.

## Project Structure

```
fastapi_app/
├── main.py                 # Application entry point
├── api/                    # API routes and dependencies
├── core/                   # Configuration and security
├── models/                 # Pydantic models
├── services/               # Business logic
├── db/                     # Database connections
└── utils/                  # Utility functions
```

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Download spaCy model:

```bash
python -m spacy download en_core_web_sm
```

3. Download dlib face landmark predictor (optional):

```bash
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
mv shape_predictor_68_face_landmarks.dat /app/
```

4. Create `.env` file:

```bash
cp .env.example .env
# Edit .env with your configuration
```

## Running the Application

### Development

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 3304
```

### Production

```bash
uvicorn main:app --host 0.0.0.0 --port 3304 --workers 4
```

## API Documentation

Once running, visit:

- Swagger UI: http://localhost:3304/docs
- ReDoc: http://localhost:3304/redoc

## Key Features

- **Feature Groups**: Manage and process feature groups with Spark
- **Training Datasets**: Create and manage training datasets
- **Privacy Detection**: Detect PII in data and faces/eyes in images
- **Data Analytics**: Word frequency analysis across datasets
- **CSV Import**: Import and transform CSV files to Hive tables

## Endpoints

### Feature Groups

- `POST /api/v1/feature-group-data` - Save feature group data
- `POST /api/v1/preview-feature-group-data` - Preview feature group
- `POST /api/v1/download-feature-group-data` - Download as CSV
- `POST /api/v1/delete-feature-group-data` - Delete feature group
- `POST /api/v1/feature-group-size/` - Get storage size

### Training Datasets

- `POST /api/v1/training-dataset-data` - Save training dataset
- `POST /api/v1/preview-training-dataset-data` - Preview dataset
- `POST /api/v1/delete-training-dataset-data` - Delete dataset

### Analytics

- `GET /api/v1/dataset-word-count` - Dataset name word frequency
- `GET /api/v1/featurestore-word-count` - Feature store word frequency
- `GET /api/v1/name-notebook-word-count` - Notebook name word frequency
- `GET /api/v1/filemanagement-word-count` - File management word frequency
- `GET /api/v1/model-zoo-word-count` - Model zoo word frequency

### Privacy

- `POST /api/v1/privacy-detection/{file_item_id}` - Detect PII/faces

### Imports

- `POST /api/v1/import-csv` - Import CSV to new Hive table
- `POST /api/v1/insert-csv` - Insert CSV to existing table
- `POST /api/v1/savtocsv` - Convert SPSS .sav to CSV

## Environment Variables

See `.env.example` for all configuration options.

## License

MIT
