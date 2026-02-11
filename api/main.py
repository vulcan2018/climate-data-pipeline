"""
ARCO Data Service - FastAPI Application

Example FastAPI application demonstrating the API structure
for ARCO data access services.

Author: S. Kalogerakos
Company: FIRA Software Ltd
Date: February 2026
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# Configuration
DATA_LAKE_PATH = os.getenv("DATA_LAKE_PATH", "s3://arco-data-lake")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Prometheus metrics
REQUEST_COUNT = Counter(
    "arco_requests_total",
    "Total number of requests",
    ["method", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "arco_request_latency_seconds",
    "Request latency in seconds",
    ["method", "endpoint"]
)


# Pydantic models
class HealthResponse(BaseModel):
    status: str
    version: str
    data_lake_path: str


class DatasetInfo(BaseModel):
    id: str
    name: str
    description: str
    format: str = "zarr"
    temporal_extent: list[str]
    spatial_extent: list[float]
    variables: list[str]


class DataQuery(BaseModel):
    dataset_id: str
    variable: str
    time_start: Optional[str] = None
    time_end: Optional[str] = None
    bbox: Optional[list[float]] = Field(
        None,
        description="Bounding box [west, south, east, north]"
    )


class DataResponse(BaseModel):
    dataset_id: str
    variable: str
    shape: list[int]
    dtype: str
    zarr_url: str
    access_latency_ms: float


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print(f"Starting ARCO Service with DATA_LAKE_PATH={DATA_LAKE_PATH}")
    yield
    # Shutdown
    print("Shutting down ARCO Service")


# FastAPI application
app = FastAPI(
    title="ARCO Data Service",
    description="""
    Analysis-Ready, Cloud-Optimised (ARCO) Data Service API.

    Provides access to Zarr-formatted climate datasets with
    optimised chunking for low-latency access.

    Part of the ECMWF Data Stores Service (DSS) infrastructure.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)


# Health endpoints
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint for Kubernetes liveness probe.
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        data_lake_path=DATA_LAKE_PATH
    )


@app.get("/ready", tags=["Health"])
async def readiness_check():
    """
    Readiness check endpoint for Kubernetes readiness probe.

    Verifies that the service can access the data lake.
    """
    # In production, would check data lake connectivity
    return {"status": "ready"}


# Metrics endpoint
@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """
    Prometheus metrics endpoint.
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


# Dataset endpoints
@app.get("/datasets", response_model=list[DatasetInfo], tags=["Datasets"])
async def list_datasets(
    collection: Optional[str] = Query(None, description="Filter by collection")
):
    """
    List available ARCO datasets.

    Returns metadata about all available datasets including
    temporal and spatial extent.
    """
    # Example response - would query STAC catalog in production
    datasets = [
        DatasetInfo(
            id="era5-2m-temperature",
            name="ERA5 2m Temperature",
            description="ECMWF ERA5 reanalysis 2m temperature",
            temporal_extent=["1979-01-01", "2024-12-31"],
            spatial_extent=[-180, -90, 180, 90],
            variables=["t2m"]
        ),
        DatasetInfo(
            id="era5-precipitation",
            name="ERA5 Total Precipitation",
            description="ECMWF ERA5 reanalysis total precipitation",
            temporal_extent=["1979-01-01", "2024-12-31"],
            spatial_extent=[-180, -90, 180, 90],
            variables=["tp"]
        ),
    ]

    REQUEST_COUNT.labels(method="GET", endpoint="/datasets", status="200").inc()
    return datasets


@app.get("/datasets/{dataset_id}", response_model=DatasetInfo, tags=["Datasets"])
async def get_dataset(dataset_id: str):
    """
    Get metadata for a specific dataset.
    """
    # Example - would query STAC catalog
    if dataset_id == "era5-2m-temperature":
        return DatasetInfo(
            id="era5-2m-temperature",
            name="ERA5 2m Temperature",
            description="ECMWF ERA5 reanalysis 2m temperature",
            temporal_extent=["1979-01-01", "2024-12-31"],
            spatial_extent=[-180, -90, 180, 90],
            variables=["t2m"]
        )

    raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")


# Data access endpoints
@app.post("/data/query", response_model=DataResponse, tags=["Data Access"])
async def query_data(query: DataQuery):
    """
    Query ARCO data and return Zarr access URL.

    This endpoint validates the query and returns the Zarr store
    URL for direct client access. The actual data is accessed
    via Zarr protocol, not through this API.
    """
    start_time = time.time()

    # Construct Zarr URL
    zarr_url = f"{DATA_LAKE_PATH}/{query.dataset_id}/{query.variable}.zarr"

    # Simulate some processing
    # In production, would validate dataset exists and query parameters

    latency_ms = (time.time() - start_time) * 1000

    REQUEST_LATENCY.labels(method="POST", endpoint="/data/query").observe(latency_ms / 1000)
    REQUEST_COUNT.labels(method="POST", endpoint="/data/query", status="200").inc()

    return DataResponse(
        dataset_id=query.dataset_id,
        variable=query.variable,
        shape=[365, 180, 360],  # Example shape
        dtype="float32",
        zarr_url=zarr_url,
        access_latency_ms=round(latency_ms, 2)
    )


@app.get("/data/{dataset_id}/{variable}/metadata", tags=["Data Access"])
async def get_variable_metadata(dataset_id: str, variable: str):
    """
    Get metadata for a specific variable within a dataset.

    Returns chunking information, compression, and statistics.
    """
    return {
        "dataset_id": dataset_id,
        "variable": variable,
        "dimensions": ["time", "latitude", "longitude"],
        "shape": [365, 180, 360],
        "chunks": [50, 50, 50],
        "dtype": "float32",
        "compression": "zstd",
        "compression_level": 3,
        "statistics": {
            "min": -40.0,
            "max": 50.0,
            "mean": 15.2,
            "std": 12.5
        }
    }


# STAC endpoints
@app.get("/stac", tags=["STAC"])
async def stac_root():
    """
    STAC Catalog root endpoint.
    """
    return {
        "type": "Catalog",
        "stac_version": "1.0.0",
        "id": "arco-catalog",
        "title": "ARCO Data Catalog",
        "description": "Analysis-Ready Cloud-Optimised climate data catalog",
        "links": [
            {"rel": "self", "href": "/stac"},
            {"rel": "root", "href": "/stac"},
            {"rel": "child", "href": "/stac/collections"}
        ]
    }


@app.get("/stac/collections", tags=["STAC"])
async def stac_collections():
    """
    List STAC collections.
    """
    return {
        "collections": [
            {
                "type": "Collection",
                "stac_version": "1.0.0",
                "id": "era5",
                "title": "ERA5 Reanalysis",
                "description": "ECMWF ERA5 global reanalysis"
            }
        ],
        "links": [
            {"rel": "self", "href": "/stac/collections"},
            {"rel": "root", "href": "/stac"}
        ]
    }


@app.get("/stac/search", tags=["STAC"])
async def stac_search(
    bbox: Optional[str] = Query(None, description="Bounding box (west,south,east,north)"),
    datetime: Optional[str] = Query(None, description="Date range (start/end)"),
    collections: Optional[str] = Query(None, description="Collection IDs (comma-separated)")
):
    """
    STAC search endpoint.

    Search for items matching spatial and temporal criteria.
    """
    return {
        "type": "FeatureCollection",
        "features": [],
        "links": [
            {"rel": "self", "href": "/stac/search"},
            {"rel": "root", "href": "/stac"}
        ],
        "context": {
            "returned": 0,
            "matched": 0
        }
    }


# Error handlers
@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    """Handle unexpected errors."""
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status="500"
    ).inc()

    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
