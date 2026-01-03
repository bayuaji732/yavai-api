from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import feature_groups, training_datasets, datasets, privacy, imports, dataprep

app = FastAPI(title="YavAI API", version="2.0.0", description="API for feature store management and data profiling")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(feature_groups.router, prefix="/api/v1", tags=["feature-groups"])
app.include_router(training_datasets.router, prefix="/api/v1", tags=["training-datasets"])
app.include_router(datasets.router, prefix="/api/v1", tags=["datasets"])
app.include_router(privacy.router, prefix="/api/v1", tags=["privacy"])
app.include_router(imports.router, prefix="/api/v1", tags=["imports"])
app.include_router(dataprep.router, prefix="/api/v1", tags=["dataprep"])

@app.get("/")
async def root():
    return {"message": "YavAI API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}