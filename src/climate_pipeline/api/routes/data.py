"""Data access endpoints for climate datasets."""

from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/api/v1/data", tags=["data"])


# Sample datasets for demonstration
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


@router.get("/datasets")
async def list_datasets() -> dict:
    """List available datasets.

    Returns:
        List of dataset metadata
    """
    return {
        "datasets": list(SAMPLE_DATASETS.values()),
        "count": len(SAMPLE_DATASETS),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/datasets/{dataset_id}")
async def get_dataset(dataset_id: str) -> dict:
    """Get detailed information about a specific dataset.

    Args:
        dataset_id: Dataset identifier

    Returns:
        Dataset metadata and available operations
    """
    if dataset_id not in SAMPLE_DATASETS:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    dataset = SAMPLE_DATASETS[dataset_id]

    return {
        **dataset,
        "available_metrics": [
            "monthly_mean",
            "seasonal_mean",
            "annual_mean",
            "climatology",
            "anomaly",
            "percentiles",
            "trend",
        ],
        "available_formats": ["json", "netcdf", "zarr"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/datasets/{dataset_id}/point")
async def get_point_data(
    dataset_id: str,
    lat: Annotated[float, Query(ge=-90, le=90, description="Latitude")],
    lon: Annotated[float, Query(ge=-180, le=180, description="Longitude")],
    start_date: Annotated[str | None, Query(description="Start date (YYYY-MM-DD)")] = None,
    end_date: Annotated[str | None, Query(description="End date (YYYY-MM-DD)")] = None,
) -> dict:
    """Extract time series data at a specific point.

    Args:
        dataset_id: Dataset identifier
        lat: Latitude (-90 to 90)
        lon: Longitude (-180 to 180)
        start_date: Optional start date
        end_date: Optional end date

    Returns:
        Time series data at the specified point
    """
    if dataset_id not in SAMPLE_DATASETS:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    # Return sample data structure
    return {
        "dataset": dataset_id,
        "location": {"lat": lat, "lon": lon},
        "time_range": {
            "start": start_date or "2020-01-01",
            "end": end_date or "2020-12-31",
        },
        "variable": SAMPLE_DATASETS[dataset_id]["variable"],
        "units": SAMPLE_DATASETS[dataset_id]["units"],
        "data": {
            "times": ["2020-01-01", "2020-02-01", "2020-03-01"],
            "values": [280.5, 282.3, 285.1],
        },
        "note": "Sample data - connect to actual data store for real values",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/datasets/{dataset_id}/region")
async def get_region_data(
    dataset_id: str,
    west: Annotated[float, Query(ge=-180, le=180, description="West bound")],
    south: Annotated[float, Query(ge=-90, le=90, description="South bound")],
    east: Annotated[float, Query(ge=-180, le=180, description="East bound")],
    north: Annotated[float, Query(ge=-90, le=90, description="North bound")],
    time: Annotated[str | None, Query(description="Time (YYYY-MM-DD)")] = None,
) -> dict:
    """Extract spatial data for a region at a specific time.

    Args:
        dataset_id: Dataset identifier
        west, south, east, north: Bounding box coordinates
        time: Time slice (default: latest)

    Returns:
        Spatial data grid for the region
    """
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


@router.get("/datasets/{dataset_id}/stats")
async def get_region_stats(
    dataset_id: str,
    west: Annotated[float, Query(ge=-180, le=180)],
    south: Annotated[float, Query(ge=-90, le=90)],
    east: Annotated[float, Query(ge=-180, le=180)],
    north: Annotated[float, Query(ge=-90, le=90)],
    start_date: Annotated[str | None, Query()] = None,
    end_date: Annotated[str | None, Query()] = None,
) -> dict:
    """Get statistics for a region and time period.

    Args:
        dataset_id: Dataset identifier
        west, south, east, north: Bounding box
        start_date, end_date: Time range

    Returns:
        Statistical summary for the region
    """
    if dataset_id not in SAMPLE_DATASETS:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    return {
        "dataset": dataset_id,
        "bbox": {"west": west, "south": south, "east": east, "north": north},
        "time_range": {
            "start": start_date or "2020-01-01",
            "end": end_date or "2020-12-31",
        },
        "statistics": {
            "mean": 285.4,
            "std": 5.2,
            "min": 270.1,
            "max": 305.8,
            "p10": 278.3,
            "p50": 285.2,
            "p90": 293.1,
        },
        "units": SAMPLE_DATASETS[dataset_id]["units"],
        "note": "Sample statistics - connect to actual data store for real values",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
