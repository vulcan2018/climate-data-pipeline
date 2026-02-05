"""FastAPI application for Climate Data Pipeline."""

from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .routes import data, health, metrics

app = FastAPI(
    title="Climate Data Pipeline API",
    description="""
REST API for accessing and processing climate data.

## Features

- **Data Access**: Query climate datasets by point, region, or time
- **Temporal Metrics**: Monthly, seasonal, and annual averages
- **Percentiles**: Climatological percentile thresholds
- **Trends**: Linear trend analysis with significance testing
- **Anomalies**: Absolute and standardized anomaly computation

## Data Sources

- ERA5 Reanalysis (ECMWF)
- CAMS Atmospheric Composition
- Satellite-derived products

## Technologies

- Xarray for multi-dimensional array processing
- Dask for parallel computation
- Zarr for cloud-optimised storage
- Redis for response caching
    """,
    version="1.0.0",
    contact={
        "name": "FIRA Software Ltd",
        "url": "https://firasoftware.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(data.router)
app.include_router(metrics.router)


@app.get("/", include_in_schema=False)
async def root() -> dict:
    """Root endpoint with API information."""
    return {
        "name": "Climate Data Pipeline API",
        "version": "1.0.0",
        "description": "REST API for climate data processing and access",
        "documentation": "/docs",
        "openapi": "/openapi.json",
        "health": "/health",
        "endpoints": {
            "datasets": "/api/v1/data/datasets",
            "metrics": "/api/v1/metrics",
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/v1/info")
async def api_info() -> dict:
    """Get API capabilities and configuration."""
    return {
        "version": "1.0.0",
        "capabilities": {
            "data_access": ["point", "region", "timeseries"],
            "temporal_metrics": ["monthly", "seasonal", "annual", "climatology"],
            "statistics": ["percentiles", "trends", "anomalies"],
            "formats": ["json", "netcdf", "zarr"],
        },
        "processing": {
            "engine": "xarray",
            "parallel": "dask",
            "storage": "zarr",
            "cache": "redis",
        },
        "limits": {
            "max_points_per_request": 10000,
            "max_time_steps": 8760,
            "cache_ttl_seconds": 3600,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception) -> JSONResponse:
    """Global exception handler for unhandled errors."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
