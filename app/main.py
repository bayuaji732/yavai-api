from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import feature_groups, training_datasets, privacy, word_count
from app.api.routes import dataprep, ingestion

app = FastAPI(title="YavAI API", version="2.0.0", description="API for feature group management and data profiling")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(feature_groups.router, prefix="/api/v2", tags=["feature-groups"])
app.include_router(training_datasets.router, prefix="/api/v2", tags=["training-datasets"])
app.include_router(word_count.router, prefix="/api/v2", tags=["word-count"])
app.include_router(privacy.router, prefix="/api/v2", tags=["privacy-detection"])
app.include_router(ingestion.router, prefix="/api/v2", tags=["ingestion"])
app.include_router(dataprep.router, prefix="/api/v2", tags=["dataprep"])

@app.get("/")
async def root():
    return {"message": "YavAI API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}