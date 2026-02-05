"""Metrics computation endpoints."""

from datetime import datetime, timezone
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/api/v1/metrics", tags=["metrics"])


@router.get("/temporal/{dataset_id}")
async def compute_temporal_metrics(
    dataset_id: str,
    metric: Annotated[
        Literal["monthly", "seasonal", "annual"],
        Query(description="Temporal aggregation type"),
    ],
    lat: Annotated[float, Query(ge=-90, le=90)],
    lon: Annotated[float, Query(ge=-180, le=180)],
    start_year: Annotated[int | None, Query(ge=1940, le=2100)] = None,
    end_year: Annotated[int | None, Query(ge=1940, le=2100)] = None,
) -> dict:
    """Compute temporal averages for a point location.

    Args:
        dataset_id: Dataset identifier
        metric: Type of temporal average
        lat, lon: Location coordinates
        start_year, end_year: Optional year range

    Returns:
        Temporal metric values
    """
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


@router.get("/percentiles/{dataset_id}")
async def compute_percentiles(
    dataset_id: str,
    lat: Annotated[float, Query(ge=-90, le=90)],
    lon: Annotated[float, Query(ge=-180, le=180)],
    percentiles: Annotated[
        str,
        Query(description="Comma-separated percentiles, e.g., '10,25,50,75,90'"),
    ] = "10,25,50,75,90,95,99",
    reference_start: Annotated[int | None, Query(ge=1940)] = None,
    reference_end: Annotated[int | None, Query(le=2100)] = None,
) -> dict:
    """Compute climatological percentiles for a location.

    Args:
        dataset_id: Dataset identifier
        lat, lon: Location coordinates
        percentiles: Percentile values to compute
        reference_start, reference_end: Reference period years

    Returns:
        Percentile thresholds by month
    """
    pct_list = [int(p.strip()) for p in percentiles.split(",")]

    # Sample percentile values by month
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


@router.get("/trend/{dataset_id}")
async def compute_trend(
    dataset_id: str,
    lat: Annotated[float, Query(ge=-90, le=90)],
    lon: Annotated[float, Query(ge=-180, le=180)],
    start_year: Annotated[int | None, Query(ge=1940)] = None,
    end_year: Annotated[int | None, Query(le=2100)] = None,
    confidence: Annotated[float, Query(ge=0.8, le=0.99)] = 0.95,
) -> dict:
    """Compute linear trend with significance testing.

    Args:
        dataset_id: Dataset identifier
        lat, lon: Location coordinates
        start_year, end_year: Period for trend calculation
        confidence: Confidence level for significance testing

    Returns:
        Trend statistics including slope, p-value, and confidence interval
    """
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


@router.get("/anomaly/{dataset_id}")
async def compute_anomaly(
    dataset_id: str,
    lat: Annotated[float, Query(ge=-90, le=90)],
    lon: Annotated[float, Query(ge=-180, le=180)],
    time: Annotated[str, Query(description="Date (YYYY-MM-DD)")],
    anomaly_type: Annotated[
        Literal["absolute", "standardized"],
        Query(description="Type of anomaly"),
    ] = "absolute",
    reference_start: Annotated[int | None, Query()] = None,
    reference_end: Annotated[int | None, Query()] = None,
) -> dict:
    """Compute anomaly for a specific time and location.

    Args:
        dataset_id: Dataset identifier
        lat, lon: Location coordinates
        time: Date for anomaly calculation
        anomaly_type: "absolute" or "standardized" (z-score)
        reference_start, reference_end: Reference period years

    Returns:
        Anomaly value with context
    """
    # Parse month for climatology lookup
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
        # For absolute anomalies, use temperature-specific thresholds
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
