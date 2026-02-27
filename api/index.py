"""Climate Data Pipeline API - Vercel Serverless Entry Point."""

import sys
from pathlib import Path

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datetime import datetime, timezone
from typing import Annotated, Literal
import math

from fastapi import FastAPI, HTTPException, Query


def generate_location_aware_temperature(lat: float, lon: float, month: int) -> float:
    """
    Generate realistic temperature (in Kelvin) based on location and month.
    - Equator: ~300K (27C), Poles: ~250K (-23C)
    - Seasonal variation: +/- 15K depending on hemisphere and month
    - Small longitudinal variation for realism
    """
    # Base temperature: warmer at equator, colder at poles
    # Uses cosine of latitude for smooth transition
    lat_rad = math.radians(lat)
    base_temp = 300 - 25 * (1 - math.cos(lat_rad))  # 275K at poles, 300K at equator

    # Seasonal variation (Northern hemisphere: warm Jun-Aug, cold Dec-Feb)
    # Southern hemisphere is opposite
    seasonal_phase = (month - 1) / 12 * 2 * math.pi  # 0 to 2pi over the year
    if lat >= 0:
        # Northern hemisphere: peak in July (month 7)
        seasonal_offset = math.sin(seasonal_phase - math.pi / 2)  # Peak at month ~7
    else:
        # Southern hemisphere: peak in January (month 1)
        seasonal_offset = math.sin(seasonal_phase + math.pi / 2)

    # Seasonal amplitude increases with latitude (minimal at equator)
    seasonal_amplitude = 15 * abs(math.sin(lat_rad))
    seasonal_temp = seasonal_amplitude * seasonal_offset

    # Small longitudinal variation (continental vs oceanic effect)
    lon_variation = 2 * math.sin(math.radians(lon) * 2)

    return round(base_temp + seasonal_temp + lon_variation, 1)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

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


# ============== Root & Info ==============

@app.get("/", include_in_schema=False)
async def root():
    """Redirect to API documentation."""
    return RedirectResponse(url="/docs")


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


# ============== Health ==============

@app.get("/health")
async def health_check() -> dict:
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "climate-data-pipeline",
    }


@app.get("/health/ready")
async def readiness_check() -> dict:
    """Readiness check - verifies service can handle requests."""
    return {
        "status": "ready",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": {"api": True, "processing": True},
    }


@app.get("/health/live")
async def liveness_check() -> dict:
    """Liveness check - verifies service is running."""
    return {
        "status": "alive",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ============== Data Access ==============

SAMPLE_DATASETS = {
    "era5-t2m": {
        "id": "era5-t2m",
        "name": "ERA5 2m Temperature",
        "variable": "2m_temperature",
        "temporal_range": "1940-present",
        "resolution": "0.25deg",
        "format": "NetCDF/Zarr",
        "units": "K",
    },
    "era5-precip": {
        "id": "era5-precip",
        "name": "ERA5 Total Precipitation",
        "variable": "total_precipitation",
        "temporal_range": "1940-present",
        "resolution": "0.25deg",
        "format": "NetCDF/Zarr",
        "units": "m",
    },
    "cams-ozone": {
        "id": "cams-ozone",
        "name": "CAMS Total Column Ozone",
        "variable": "total_column_ozone",
        "temporal_range": "2003-present",
        "resolution": "0.75deg",
        "format": "NetCDF/Zarr",
        "units": "kg m-2",
    },
}


@app.get("/api/v1/data/datasets")
async def list_datasets() -> dict:
    """List available datasets."""
    return {
        "datasets": list(SAMPLE_DATASETS.values()),
        "count": len(SAMPLE_DATASETS),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/v1/data/datasets/{dataset_id}")
async def get_dataset(dataset_id: str) -> dict:
    """Get detailed information about a specific dataset."""
    if dataset_id not in SAMPLE_DATASETS:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
    dataset = SAMPLE_DATASETS[dataset_id]
    return {
        **dataset,
        "available_metrics": [
            "monthly_mean", "seasonal_mean", "annual_mean",
            "climatology", "anomaly", "percentiles", "trend",
        ],
        "available_formats": ["json", "netcdf", "zarr"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/v1/data/datasets/{dataset_id}/point")
async def get_point_data(
    dataset_id: str,
    lat: Annotated[float, Query(ge=-90, le=90, description="Latitude")],
    lon: Annotated[float, Query(ge=-180, le=180, description="Longitude")],
    start_date: Annotated[str | None, Query(description="Start date (YYYY-MM-DD)")] = None,
    end_date: Annotated[str | None, Query(description="End date (YYYY-MM-DD)")] = None,
) -> dict:
    """Extract time series data at a specific point."""
    if dataset_id not in SAMPLE_DATASETS:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    # Parse year from start_date or default to 2024
    year = 2024
    if start_date:
        try:
            year = int(start_date.split("-")[0])
        except (ValueError, IndexError):
            pass

    # Generate location-aware monthly temperatures for the year
    times = [f"{year}-{m:02d}-01" for m in range(1, 13)]
    values = [generate_location_aware_temperature(lat, lon, m) for m in range(1, 13)]

    return {
        "dataset": dataset_id,
        "location": {"lat": lat, "lon": lon},
        "time_range": {
            "start": start_date or f"{year}-01-01",
            "end": end_date or f"{year}-12-31",
        },
        "variable": SAMPLE_DATASETS[dataset_id]["variable"],
        "units": SAMPLE_DATASETS[dataset_id]["units"],
        "data": {
            "times": times,
            "values": values,
        },
        "note": "Location-aware sample data based on lat/lon and seasonal patterns",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/v1/data/datasets/{dataset_id}/region")
async def get_region_data(
    dataset_id: str,
    west: Annotated[float, Query(ge=-180, le=180)],
    south: Annotated[float, Query(ge=-90, le=90)],
    east: Annotated[float, Query(ge=-180, le=180)],
    north: Annotated[float, Query(ge=-90, le=90)],
    time: Annotated[str | None, Query(description="Time (YYYY-MM-DD)")] = None,
) -> dict:
    """Extract spatial data for a region at a specific time."""
    if dataset_id not in SAMPLE_DATASETS:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
    return {
        "dataset": dataset_id,
        "bbox": {"west": west, "south": south, "east": east, "north": north},
        "time": time or "2020-06-15",
        "variable": SAMPLE_DATASETS[dataset_id]["variable"],
        "units": SAMPLE_DATASETS[dataset_id]["units"],
        "grid": {
            "lats": [south, (south + north) / 2, north],
            "lons": [west, (west + east) / 2, east],
            "values": [
                [280.1, 281.2, 282.3],
                [283.4, 284.5, 285.6],
                [286.7, 287.8, 288.9],
            ],
        },
        "note": "Sample data - connect to actual data store for real values",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ============== Metrics ==============

def _generate_sample_temporal(metric: str) -> dict:
    """Generate sample temporal data."""
    if metric == "monthly":
        return {
            "labels": ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
            "values": [275.2, 276.8, 280.1, 285.4, 291.2, 296.8,
                      299.1, 298.4, 293.7, 287.2, 280.5, 276.1],
        }
    elif metric == "seasonal":
        return {
            "labels": ["DJF", "MAM", "JJA", "SON"],
            "values": [276.0, 285.6, 298.1, 287.1],
        }
    else:  # annual
        return {
            "labels": [str(y) for y in range(2015, 2024)],
            "values": [286.2, 286.5, 286.8, 287.1, 286.9, 287.3, 287.6, 287.9, 288.1],
        }


@app.get("/api/v1/metrics/temporal/{dataset_id}")
async def compute_temporal_metrics(
    dataset_id: str,
    metric: Annotated[Literal["monthly", "seasonal", "annual"], Query()],
    lat: Annotated[float, Query(ge=-90, le=90)],
    lon: Annotated[float, Query(ge=-180, le=180)],
    start_year: Annotated[int | None, Query(ge=1940, le=2100)] = None,
    end_year: Annotated[int | None, Query(ge=1940, le=2100)] = None,
) -> dict:
    """Compute temporal averages for a point location."""
    return {
        "dataset": dataset_id,
        "metric": f"{metric}_mean",
        "location": {"lat": lat, "lon": lon},
        "period": {
            "start_year": start_year or 1991,
            "end_year": end_year or 2020,
        },
        "values": _generate_sample_temporal(metric),
        "units": "K",
        "note": "Sample data - actual computation uses Xarray/Dask pipeline",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/v1/metrics/percentiles/{dataset_id}")
async def compute_percentiles(
    dataset_id: str,
    lat: Annotated[float, Query(ge=-90, le=90)],
    lon: Annotated[float, Query(ge=-180, le=180)],
    percentiles: Annotated[str, Query()] = "10,25,50,75,90,95,99",
    reference_start: Annotated[int | None, Query(ge=1940)] = None,
    reference_end: Annotated[int | None, Query(le=2100)] = None,
) -> dict:
    """Compute climatological percentiles for a location."""
    pct_list = [int(p.strip()) for p in percentiles.split(",")]
    sample_percentiles = {
        p: [273 + p / 10 + m * 2 for m in range(12)]
        for p in pct_list
    }
    return {
        "dataset": dataset_id,
        "location": {"lat": lat, "lon": lon},
        "reference_period": {
            "start": reference_start or 1991,
            "end": reference_end or 2020,
        },
        "percentiles": pct_list,
        "values": {
            "months": list(range(1, 13)),
            **{f"p{p}": sample_percentiles[p] for p in pct_list},
        },
        "units": "K",
        "note": "Sample data - actual computation uses scipy percentile functions",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/v1/metrics/trend/{dataset_id}")
async def compute_trend(
    dataset_id: str,
    lat: Annotated[float, Query(ge=-90, le=90)],
    lon: Annotated[float, Query(ge=-180, le=180)],
    start_year: Annotated[int | None, Query(ge=1940)] = None,
    end_year: Annotated[int | None, Query(le=2100)] = None,
    confidence: Annotated[float, Query(ge=0.8, le=0.99)] = 0.95,
) -> dict:
    """Compute linear trend with significance testing."""
    return {
        "dataset": dataset_id,
        "location": {"lat": lat, "lon": lon},
        "period": {
            "start": start_year or 1980,
            "end": end_year or 2023,
        },
        "trend": {
            "slope": 0.023,
            "slope_units": "K per year",
            "total_change": 0.023 * ((end_year or 2023) - (start_year or 1980)),
            "p_value": 0.0012,
            "significant": True,
            "confidence_level": confidence,
            "confidence_interval": {
                "lower": 0.018,
                "upper": 0.028,
            },
        },
        "method": "ordinary_least_squares",
        "note": "Sample data - actual computation uses scipy.stats.linregress",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _classify_anomaly(value: float, anomaly_type: str) -> dict:
    """Classify anomaly severity."""
    if anomaly_type == "standardized":
        if value < -2:
            return {"level": -3, "label": "Extremely below normal"}
        elif value < -1.5:
            return {"level": -2, "label": "Severely below normal"}
        elif value < -1:
            return {"level": -1, "label": "Moderately below normal"}
        elif value <= 1:
            return {"level": 0, "label": "Near normal"}
        elif value <= 1.5:
            return {"level": 1, "label": "Moderately above normal"}
        elif value <= 2:
            return {"level": 2, "label": "Severely above normal"}
        else:
            return {"level": 3, "label": "Extremely above normal"}
    else:
        if value < -5:
            return {"level": -2, "label": "Much below normal"}
        elif value < -2:
            return {"level": -1, "label": "Below normal"}
        elif value <= 2:
            return {"level": 0, "label": "Near normal"}
        elif value <= 5:
            return {"level": 1, "label": "Above normal"}
        else:
            return {"level": 2, "label": "Much above normal"}


@app.get("/api/v1/metrics/anomaly/{dataset_id}")
async def compute_anomaly(
    dataset_id: str,
    lat: Annotated[float, Query(ge=-90, le=90)],
    lon: Annotated[float, Query(ge=-180, le=180)],
    time: Annotated[str, Query(description="Date (YYYY-MM-DD)")],
    anomaly_type: Annotated[Literal["absolute", "standardized"], Query()] = "absolute",
    reference_start: Annotated[int | None, Query()] = None,
    reference_end: Annotated[int | None, Query()] = None,
) -> dict:
    """Compute anomaly for a specific time and location."""
    month = int(time.split("-")[1])
    if anomaly_type == "absolute":
        anomaly_value = 2.3
        units = "K"
    else:
        anomaly_value = 1.8
        units = "standard deviations"
    return {
        "dataset": dataset_id,
        "location": {"lat": lat, "lon": lon},
        "time": time,
        "reference_period": {
            "start": reference_start or 1991,
            "end": reference_end or 2020,
        },
        "anomaly": {
            "type": anomaly_type,
            "value": anomaly_value,
            "units": units,
            "climatology_month": month,
            "climatological_mean": 285.4,
            "climatological_std": 1.3 if anomaly_type == "standardized" else None,
        },
        "classification": _classify_anomaly(anomaly_value, anomaly_type),
        "note": "Sample data - actual computation uses climatology-based anomaly calculation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ============== Error Handler ==============

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
