from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import feature_groups, training_datasets, datasets, privacy, imports

app = FastAPI(title="YavAI Feature Store API", version="1.0.0")

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

@app.get("/")
async def root():
    return {"message": "YavAI Feature Store API"}